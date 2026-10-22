import argparse
from scipy.spatial.transform import Rotation
from common import ROOT, np, tensor, torch, assert_open
from marker_model import MarkerHand
from io_utils import save, stats


def acceleration(values, times):
    intervals = np.diff(times)
    velocity = np.diff(values, axis=0) / intervals.reshape((-1,) + (1,) * (values.ndim - 1))
    result = 2 * np.diff(velocity, axis=0) / (intervals[1:] + intervals[:-1]).reshape((-1,) + (1,) * (values.ndim - 1))
    return result, (intervals[1:] < .12) & (intervals[:-1] < .12)


def calculate(fit, model):
    times = fit['timestamps_seconds']
    rotation = Rotation.from_rotvec(fit['orient']).as_matrix()
    local = np.einsum('bji,bvj->bvi', rotation, fit['vertices_world_m'] - fit['wrist'][:, None])
    center = (model.regressor @ (model.template + torch.einsum('k,vck->vc', tensor(fit['beta']), model.shape)))[0].numpy()
    template = model.template.numpy() - center
    root_ids = np.flatnonzero((model.weights[:, 0].numpy() > .9) & (np.linalg.norm(template, axis=-1) < .045))
    palm_ids = np.flatnonzero((model.weights[:, 0].numpy() > .7) & (np.linalg.norm(template, axis=-1) < .08))
    thumb_rotations = Rotation.from_rotvec(fit['pose'].reshape(-1, 15, 3)[:, 12:].reshape(-1, 3)).as_matrix().reshape(-1, 3, 3, 3)
    steps = Rotation.from_matrix((thumb_rotations[:-1].transpose(0, 1, 3, 2) @ thumb_rotations[1:]).reshape(-1, 3, 3)).magnitude().reshape(-1, 3)
    keep = np.diff(times) < .12
    root_acceleration, continuous = acceleration(fit['wrist'], times)
    root_local_acceleration, unused = acceleration(local[:, root_ids], times)
    palm_acceleration, unused = acceleration(local[:, palm_ids], times)
    quaternion = Rotation.from_matrix(thumb_rotations.reshape(-1, 3, 3)).as_quat().reshape(-1, 3, 4)
    twist = 2 * np.arctan2((quaternion[..., :3] * model.axes.numpy()[None, 12:]).sum(-1), quaternion[..., 3])
    twist = np.arctan2(np.sin(twist), np.cos(twist))
    pose = fit['pose'].copy().reshape(-1, 15, 3)
    pose[:, 12:] = np.median(pose[:, 12:], axis=0)
    with torch.no_grad():
        neutral_thumb, unused_joints = model.forward(tensor(fit['beta']), tensor(pose.reshape(-1, 45)),
            torch.zeros(len(pose), 3), torch.zeros(len(pose), 3), vertices=True)
    fixed_thumb_acceleration, unused = acceleration(neutral_thumb.numpy()[:, root_ids], times)
    effect = local[:, root_ids] - neutral_thumb.numpy()[:, root_ids]
    effect_acceleration, unused = acceleration(effect, times)
    worst = np.argsort(np.where(keep, steps.max(-1), -1))[-12:][::-1]
    return dict(root_ids=root_ids.tolist(), palm_ids=palm_ids.tolist(),
        wrist_acceleration_m_s2=stats(np.linalg.norm(root_acceleration[continuous], axis=-1)),
        palm_root_local_acceleration_m_s2=stats(np.linalg.norm(root_local_acceleration[continuous], axis=-1)),
        palm_local_acceleration_m_s2=stats(np.linalg.norm(palm_acceleration[continuous], axis=-1)),
        palm_root_fixed_thumb_acceleration_m_s2=stats(np.linalg.norm(fixed_thumb_acceleration[continuous], axis=-1)),
        thumb_contribution_palm_root_acceleration_m_s2=stats(np.linalg.norm(effect_acceleration[continuous], axis=-1)),
        thumb_local_rotation_step_degrees=stats(np.degrees(steps[keep])),
        thumb_axial_twist_degrees=stats(np.degrees(abs(twist))),
        thumb_axial_twist_by_joint_degrees={name:stats(np.degrees(abs(twist[:, index]))) for index,name in enumerate(['CMC', 'MCP', 'IP'])},
        worst_thumb_steps=[dict(index=int(index), times_seconds=times[index:index + 2].tolist(), degrees=np.degrees(steps[index]).tolist()) for index in worst],
        root_patch_definition='MANO root skinning weight >0.9 and neutral wrist distance <45mm; shared across candidates',
        bone_axes_are_template_axes_not_measured_anatomy=True,
        fixed_thumb_is_diagnostic_counterfactual_not_candidate=True)


def main():
    assert_open()
    parser = argparse.ArgumentParser()
    parser.add_argument('--fits', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    model = MarkerHand(np.zeros((14, 3)), 'left')
    base = dict(np.load(ROOT / 'baseline/left_fit.npz'))
    model.set_shape_axes(base['beta'])
    results = {name:calculate(dict(np.load(ROOT / name / 'left_fit.npz')), model) for name in args.fits}
    save(ROOT / args.output, results)
    for name, result in results.items():
        print(name, {key:result[key] for key in ['palm_root_local_acceleration_m_s2', 'palm_root_fixed_thumb_acceleration_m_s2', 'thumb_contribution_palm_root_acceleration_m_s2', 'thumb_local_rotation_step_degrees']}, flush=True)


if __name__ == '__main__':
    main()
