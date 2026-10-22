import ast
import json
from common import ROOT, np, assert_open
from io_utils import save, digest


def main():
    assert_open()
    selection = json.loads((ROOT / 'selection.json').read_text())
    gate = json.loads((ROOT / selection['gate']).read_text())
    base = dict(np.load(ROOT / 'baseline/left_fit.npz'))
    fit = dict(np.load(ROOT / 'selected_review/left_fit.npz'))
    checks = dict(accepted_gate=gate['passed'], fixed_other_fingers=np.array_equal(base['pose'][:, :36], fit['pose'][:, :36]),
        selected_candidate_hash=digest(ROOT / 'selected_review/left_fit.npz') == selection['candidate_sha256'],
        right_baseline_hash=digest(ROOT / 'baseline/right_fit.npz') == digest(ROOT / 'selected_review/right_fit.npz'))
    for key in ['wrist', 'orient', 'beta', 'marker_height_m', 'marker_vertex_ids',
            'marker_attachment_faces', 'marker_attachment_barycentric', 'wrist_marker_local_offsets_m',
            'wrist_articulation_matrix', 'timestamps_seconds', 'rgb_indices']:
        checks['fixed.' + key] = bool(np.array_equal(base[key], fit[key]))
    for side in ['left', 'right']:
        checks['observations.' + side] = digest(ROOT / 'inputs_v2' / (side + '.npz')) == digest(ROOT / 'selected_inputs' / (side + '.npz'))
    for name in json.loads((ROOT / 'code_manifest.json').read_text())['python_files']:
        ast.parse((ROOT / name).read_text())
        checks['AST.' + name] = True
    active = fit['stability_adopted_frames']
    checks['inactive_pose_exact'] = bool(np.array_equal(base['pose'][~active], fit['pose'][~active]))
    save(ROOT / 'stability_verification.json', dict(passed=all(checks.values()), checks=checks,
        active_samples=int(active.sum()), total_samples=len(active), independent_truth=False))
    print('VERIFIED', all(checks.values()), len(checks), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
