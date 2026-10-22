import argparse
import json
from common import ROOT, np, assert_open
from io_utils import save, digest


def main():
    assert_open()
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--output')
    parser.add_argument('--visual-review')
    args = parser.parse_args()
    candidate = args.candidate
    checks = []

    def add(name, before, after, tolerance=0, larger=False):
        passed = after >= before - tolerance if larger else after <= before + tolerance
        checks.append(dict(metric=name, baseline=float(before), candidate=float(after),
            tolerance=tolerance, larger_is_better=larger, passed=bool(passed)))

    def compare(name, first, last, tolerance, larger=False):
        add(name + '.count', first['count'], last['count'], larger=True)
        add(name + '.count_reverse', last['count'], first['count'], larger=True)
        for key in ['median', 'p95', 'maximum']:
            if key in first:
                add(name + '.' + key, first[key], last[key], tolerance, larger)

    metrics = json.loads((ROOT / (args.prefix + '_metrics.json')).read_text())['hands']
    for side, collection in metrics.items():
        first, last = collection['baseline'], collection[candidate]
        compare(side + '.native', first['native_all_mm'], last['native_all_mm'], .001)
        for region, value in first['native_regions_mm'].items():
            compare(side + '.native.' + region, value, last['native_regions_mm'][region], .001)
        for first_clip, last_clip in zip(first['clips'], last['clips']):
            for region in ['all_target_mm', 'hand_surface_mm', 'wrist_mm', 'distal_mm']:
                compare(side + '.clip' + str(first_clip['start']) + '.' + region, first_clip[region], last_clip[region], .001)
            add(side + '.clip' + str(first_clip['start']) + '.compensation', first_clip['compensatory_finger_frames'], last_clip['compensatory_finger_frames'])
        for key in ['wrist_acceleration_m_s2', 'all_joint_acceleration_m_s2']:
            compare(side + '.' + key, first[key], last[key], 1e-4)
        for key in ['over_12mm', 'negative_PIP_positive_DIP_count']:
            add(side + '.' + key, first[key], last[key])
    finger = json.loads((ROOT / (args.prefix + '_finger/result.json')).read_text())
    expected = json.loads((ROOT / 'evaluation_finger_records.json').read_text())
    identity = lambda row:(row['rgb'], row['finger'], row['fraction'])
    add('frozen_finger_record_identity', 1, int({identity(row) for row in finger['records']} == {identity(row) for row in expected['records']}), larger=True)
    for group, collection in finger['summary'].items():
        for key in ['center_abs_px', 'edge_abs_px']:
            compare('finger.' + group + '.' + key, collection['baseline'][key], collection[candidate][key], 1e-4)
    silhouettes = json.loads((ROOT / (args.prefix + '_silhouette/result.json')).read_text())
    for group, collection in silhouettes['summary'].items():
        for key in ['contour_median_px', 'contour_p95_px', 'iou']:
            compare('silhouette.' + group + '.' + key, collection['baseline'][key], collection[candidate][key],
                1e-6 if key == 'iou' else 1e-4, larger=key == 'iou')
    side = json.loads((ROOT / (args.prefix + '_side/result.json')).read_text())['summary']
    compare('side.contour', side['baseline'], side[candidate], 1e-4)
    stability = json.loads((ROOT / (args.prefix + '_stability.json')).read_text())
    first, last = stability['baseline'], stability[candidate]
    for key, tolerance in [('palm_root_local_acceleration_m_s2', 1e-5), ('thumb_local_rotation_step_degrees', .001)]:
        compare(key, first[key], last[key], tolerance)
    for joint in ['MCP', 'IP']:
        compare('thumb_twist.' + joint, first['thumb_axial_twist_by_joint_degrees'][joint], last['thumb_axial_twist_by_joint_degrees'][joint], .001)
    base = dict(np.load(ROOT / 'baseline/left_fit.npz'))
    fit = dict(np.load(ROOT / candidate / 'left_fit.npz'))
    for key in ['beta', 'orient', 'wrist', 'marker_height_m', 'marker_vertex_ids',
            'wrist_marker_local_offsets_m', 'marker_attachment_faces', 'marker_attachment_barycentric', 'wrist_articulation_matrix']:
        add('fixed.' + key, 1, int(np.array_equal(base[key], fit[key])), larger=True)
    add('fixed.other_finger_pose', 1, int(np.array_equal(base['pose'][:, :36], fit['pose'][:, :36])), larger=True)
    improvement = last['palm_root_local_acceleration_m_s2']['p95'] < first['palm_root_local_acceleration_m_s2']['p95'] - 1e-5
    improvement = improvement and last['thumb_local_rotation_step_degrees']['p95'] < first['thumb_local_rotation_step_degrees']['p95'] - .001
    failed = [row for row in checks if not row['passed']]
    visual_checks = [row for row in checks if row['metric'].startswith(('finger.', 'silhouette.', 'side.'))]
    hard_failed = [row for row in failed if row not in visual_checks]
    for row in visual_checks:
        allowance = .003 if '.iou.' in row['metric'] else 0 if '.count' in row['metric'] else .5
        row['small_visual_proxy_allowance'] = allowance
        row['within_visual_proxy_allowance'] = bool(row['candidate'] >= row['baseline'] - allowance
            if row['larger_is_better'] else row['candidate'] <= row['baseline'] + allowance)
    visual_failed = [row for row in visual_checks if not row['within_visual_proxy_allowance']]
    reviewed = False
    if args.visual_review:
        record = json.loads((ROOT / args.visual_review).read_text())
        reviewed = (record['candidate_sha256'] == digest(ROOT / candidate / 'left_fit.npz')
            and record['no_obvious_added_deformation']
            and set(row['metric'] for row in visual_failed).issubset(set(record['reviewed_visual_proxy_flags'])))
    numerical_passed = not hard_failed and (not visual_failed or reviewed) and improvement
    save(ROOT / (args.output if args.output else args.prefix + '_' + candidate + '_gate.json'), dict(candidate=candidate, passed=numerical_passed,
        check_count=len(checks), strict_nonregression_passed=not failed, hard_nonregression_passed=not hard_failed,
        small_visual_proxy_change_accepted=not visual_failed, target_improvement=improvement,
        actual_visual_review_accepted=reviewed, visual_review_record=args.visual_review,
        checks=checks, failed=failed, hard_failed=hard_failed, visual_allowance_failed=visual_failed,
        user_acceptance='2026-10-22: very small contour regression allowed if no obvious visual degradation',
        visual_review_required=True, numerical_gate_is_not_visual_certification=True,
        visual_proxy_sanity_limits=dict(pixel=.5, IoU=.003),
        development_proxy_gate_not_independent_truth=True,
        numerical_tolerances=dict(marker_mm=.001, dynamics_m_s2=.0001, local_palm_dynamics_m_s2=.00001,
            angles_degrees=.001, pixel=.0001, IoU=.000001),
        no_per_frame_RGB_pose_selection=True, global_candidate_acceptance_only=True))
    print('GATE', candidate, 'passed', numerical_passed, 'hard_failed', len(hard_failed),
        'visual_allowance_failed', len(visual_failed), 'strict_failed', len(failed), 'checks', len(checks), flush=True)
    for row in failed[:12]:
        print(row, flush=True)


if __name__ == '__main__':
    main()
