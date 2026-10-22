import argparse
import shutil
from scipy.spatial.transform import Rotation
from common import ROOT, np, assert_open
from io_utils import save
from stability_metrics import acceleration


def samples(fit, root_ids):
    times = fit['timestamps_seconds']
    continuous = (np.diff(times)[1:] < .12) & (np.diff(times)[:-1] < .12)
    middle = np.arange(1, len(times) - 1)[continuous]
    rotation = Rotation.from_rotvec(fit['orient']).as_matrix()
    local = np.einsum('bji,bvj->bvi', rotation, fit['vertices_world_m'][:, root_ids] - fit['wrist'][:, None])
    local_acceleration, unused = acceleration(local, times)
    joint_acceleration, unused = acceleration(fit['joints_world_m'], times)
    pose = fit['pose'].reshape(-1, 15, 3)[:, 12:]
    rotations = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(-1, 3, 3, 3)
    keep = np.diff(times) < .12
    steps = np.degrees(Rotation.from_matrix((rotations[:-1].transpose(0, 1, 3, 2) @ rotations[1:]).reshape(-1, 3, 3)).magnitude().reshape(-1, 3))[keep]
    from marker_model import MarkerHand
    from common import tensor
    if not hasattr(samples, 'axes'):
        model = MarkerHand(np.zeros((14, 3)), 'left')
        model.set_shape_axes(fit['beta'])
        samples.axes = model.axes.numpy()[12:]
    quaternion = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat().reshape(-1, 3, 4)
    twist = 2 * np.arctan2((quaternion[..., :3] * samples.axes[None]).sum(-1), quaternion[..., 3])
    twist = np.degrees(abs(np.arctan2(np.sin(twist), np.cos(twist))))
    return dict(joint_acceleration=(np.linalg.norm(joint_acceleration[continuous], axis=-1).reshape(-1), np.repeat(middle, 16), 1, 1e-4),
        palm_root_local_acceleration=(np.linalg.norm(local_acceleration[continuous], axis=-1).reshape(-1), np.repeat(middle, len(root_ids)), 1, 1e-5),
        thumb_rotation_step=(steps.reshape(-1), np.repeat(np.flatnonzero(keep), 3), 2, .001),
        thumb_MCP_twist=(twist[:, 1], np.arange(len(times)), 0, .001),
        thumb_IP_twist=(twist[:, 2], np.arange(len(times)), 0, .001))


def blend(base, candidate, active):
    result = dict(base)
    for key in ['pose', 'vertices_world_m', 'joints_world_m', 'markers_predicted_world_m']:
        result[key] = np.where(active.reshape((-1,) + (1,) * (base[key].ndim - 1)), candidate[key], base[key])
    return result


def quantify(base_features, candidate_features):
    checks = []
    violation = 0.
    for name, (values, unused_frames, unused_context, tolerance) in base_features.items():
        proposal = candidate_features[name][0]
        for quantile, label in [(.5, 'median'), (.95, 'p95'), (1., 'maximum')]:
            before = float(np.quantile(values, quantile))
            after = float(np.quantile(proposal, quantile))
            excess = max(0., after - before - tolerance)
            scale = max(before, .01)
            violation += excess / scale
            checks.append(dict(metric=name + '.' + label, before=before, after=after,
                tolerance=tolerance, passed=excess == 0, normalized_violation=excess / scale))
    return violation, checks


def main():
    assert_open()
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(exist_ok=False)
    base = dict(np.load(ROOT / 'baseline/left_fit.npz'))
    candidate = dict(np.load(ROOT / args.candidate / 'left_fit.npz'))
    baseline_stability = __import__('json').loads((ROOT / 'baseline_stability.json').read_text())['baseline']
    root_ids = baseline_stability['root_ids']
    base_features = samples(base, root_ids)
    active = np.linalg.norm(candidate['pose'] - base['pose'], axis=-1) > 1e-7
    trace = []
    for iteration in range(len(active) + 1):
        current = blend(base, candidate, active)
        features = samples(current, root_ids)
        score, checks = quantify(base_features, features)
        if score == 0:
            break
        exposure = np.zeros(len(active))
        for row in checks:
            if row['passed']:
                continue
            name = row['metric'].rsplit('.', 1)[0]
            values, frames, context, unused_tolerance = features[name]
            baseline_values = base_features[name][0]
            contribution = np.maximum(values - baseline_values, 0) * (values > row['before']) / max(row['before'], .01)
            local = np.bincount(frames, weights=contribution, minlength=len(active))
            exposure += local
            if context == 1:
                exposure[:-1] += local[1:]
                exposure[1:] += local[:-1]
            if context == 2:
                exposure[1:] += local[:-1]
        choices = np.flatnonzero(active)
        ranked = choices[np.argsort(exposure[choices])[::-1]]
        best = None
        for frame in ranked[:32]:
            proposal = active.copy()
            proposal[frame] = False
            value, unused_checks = quantify(base_features, samples(blend(base, candidate, proposal), root_ids))
            if value < score - 1e-10 and (best is None or value < best[0]):
                best = (value, proposal, int(frame), 1)
        if best is None:
            for frame in ranked[:32]:
                proposal = active.copy()
                proposal[max(0, frame - 1):frame + 2] = False
                value, unused_checks = quantify(base_features, samples(blend(base, candidate, proposal), root_ids))
                if value < score - 1e-10 and (best is None or value < best[0]):
                    best = (value, proposal, int(frame), 3)
        if best is None:
            active[ranked[:max(1, len(ranked) // 10)]] = False
            trace.append(dict(iteration=iteration, score=score, fallback_quantile_rollback=True))
        else:
            active = best[1]
            trace.append(dict(iteration=iteration, before=score, after=best[0], rolled_back_frame=best[2], context=best[3]))
        if iteration % 5 == 0:
            print('GUARD', iteration, score, int(active.sum()), flush=True)
    result = blend(base, candidate, active)
    result['stability_adopted_frames'] = active
    np.savez_compressed(output / 'left_fit.npz', **result)
    shutil.copy2(ROOT / 'baseline/right_fit.npz', output / 'right_fit.npz')
    score, checks = quantify(base_features, samples(result, root_ids))
    save(output / 'result.json', dict(candidate=args.candidate, trace=trace, checks=checks, passed=score == 0,
        remaining_violation=score, adopted_samples=int(active.sum()), total_samples=len(active),
        selection_uses_C3D_model_dynamics_only=True, RGB_or_depth_per_frame_selection=False,
        temporal_neighbors_rechecked_after_each_rollback=True,
        method='Conservative discrete rollback of whole candidate poses; baseline fallback if no feasible improvement',
        observation_targets_fixed=True, independent_ground_truth=False))
    print('GUARD_FINAL', score, int(active.sum()), flush=True)


if __name__ == '__main__':
    main()
