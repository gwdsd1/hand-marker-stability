import argparse
import json
import shutil
from common import ROOT, assert_open
from io_utils import save, digest


def main():
    assert_open()
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--gate', required=True)
    args = parser.parse_args()
    gate = json.loads((ROOT / args.gate).read_text())
    if gate['candidate'] != args.candidate or not gate['passed']:
        raise RuntimeError('Candidate did not pass the complete acceptance policy')
    output = ROOT / 'selected_review'
    output.mkdir(exist_ok=False)
    shutil.copy2(ROOT / args.candidate / 'left_fit.npz', output / 'left_fit.npz')
    shutil.copy2(ROOT / 'baseline/right_fit.npz', output / 'right_fit.npz')
    shutil.copytree(ROOT / 'inputs_v2', ROOT / 'selected_inputs')
    save(ROOT / 'selection.json', dict(left=args.candidate, right='baseline', inputs='selected_inputs',
        gate=args.gate, candidate_sha256=digest(output / 'left_fit.npz'),
        RGB_and_depth_pose_loss=False, per_frame_RGB_selection=False,
        development_selection_not_independent_truth=True, production_default_changed=False,
        reason='Observable marker constraints retained; reduce low-observability thumb compensation and induced palm deformation',
        visual_policy='Small contour proxy changes allowed under explicit visual review'))


if __name__ == '__main__':
    main()
