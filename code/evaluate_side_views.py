import argparse
import cv2
from common import ROOT, np, torch, tensor, assert_open
from capture import Capture
from rgb_camera import RgbCamera
from marker_model import MarkerHand
from measure_silhouettes import image_mask, model_mask, score, edge
from io_utils import save, stats
from palm_view import palm_normals, facing_camera


def main():
    assert_open()
    parser = argparse.ArgumentParser()
    parser.add_argument('--fits', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference', default='baseline')
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(exist_ok=False)
    baseline = dict(np.load(ROOT / args.reference / 'left_fit.npz'))
    fits = {name: dict(np.load(ROOT / name / 'left_fit.npz')) for name in args.fits}
    capture = Capture()
    camera = RgbCamera(capture)
    rgb_rows = {int(row['rgb_index']): row for row in capture.index['rgb']}
    model = MarkerHand(np.zeros((14, 3)), 'left')
    model.set_attachment(baseline)
    normal = palm_normals(baseline)
    records = []
    previous_time = -np.inf
    for frame, time in enumerate(baseline['timestamps_seconds']):
        if time - previous_time < .18:
            continue
        row = rgb_rows[int(baseline['rgb_indices'][frame])]
        transform = camera.matrix(row)
        facing = abs(facing_camera(normal[frame],baseline['joints_world_m'][frame,0],transform))
        if facing > .35:
            continue
        vertices = baseline['vertices_world_m'][frame] @ transform[:3, :3].T + transform[:3, 3]
        pixels = cv2.projectPoints(vertices, np.zeros(3), np.zeros(3), camera.intrinsic, camera.distortion)[0][:, 0]
        lower = np.maximum(np.floor(pixels.min(0) - 35).astype(int), 0)
        upper = np.minimum(np.ceil(pixels.max(0) + 35).astype(int), [1920, 1080])
        if (upper - lower < 60).any():
            continue
        image = capture.image(row)[lower[1]:upper[1], lower[0]:upper[0]]
        size = tuple(upper - lower)
        target = image_mask(image, pixels.mean(0) - lower)
        if target is None:
            continue
        sharpness = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if sharpness < 15:
            continue
        valid = np.ones(target.shape, bool)
        valid[:5] = False
        valid[-5:] = False
        valid[:, :5] = False
        valid[:, -5:] = False
        local_joints = baseline['joints_world_m'][frame][[0, 4]] @ transform[:3, :3].T + transform[:3, 3]
        wrist_pixel, mcp_pixel = cv2.projectPoints(local_joints, np.zeros(3), np.zeros(3), camera.intrinsic, camera.distortion)[0][:, 0] - lower
        palm_direction = mcp_pixel - wrist_pixel
        palm_length = np.linalg.norm(palm_direction)
        if palm_length < 25:
            continue
        palm_direction /= palm_length
        grid_y, grid_x = np.indices(target.shape)
        valid &= (grid_x - wrist_pixel[0]) * palm_direction[0] + (grid_y - wrist_pixel[1]) * palm_direction[1] > .08 * palm_length
        interior = (cv2.erode(target, np.ones((5, 5), np.uint8)) > 0) & valid
        if interior.sum() < 150:
            continue
        laplacian = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
        hand_sharpness = float(laplacian[interior].var())
        if hand_sharpness < 3:
            continue
        scores = {}
        panels = [image.copy()]
        for name, fit in fits.items():
            predicted = model_mask(fit['vertices_world_m'][frame], fit['faces'], transform,
                camera, lower, size, 1.)
            value = score(predicted, target, valid, 1.)
            if value is not None:
                scores[name] = value
            display = image.copy()
            display[edge(target, valid)] = [0, 255, 0]
            display[edge(predicted, valid)] = [0, 120, 255]
            panels.append(display)
        if len(scores) != len(fits):
            continue
        canvas = np.full((image.shape[0] + 35, image.shape[1] * len(panels), 3), 22, np.uint8)
        for column, (name, panel) in enumerate(zip(['RGB'] + args.fits, panels)):
            canvas[35:, column * size[0]:(column + 1) * size[0]] = panel
            cv2.putText(canvas, name, (column * size[0] + 4, 23), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        path = output / f'left_RGB{row["rgb_index"]}_side.jpg'
        cv2.imwrite(str(path), canvas)
        records.append(dict(rgb=int(row['rgb_index']), time=float(time), facing=float(facing),
            sharpness=sharpness, skin_interior_sharpness=hand_sharpness, results=scores, image=path.name))
        previous_time = time
    capture.close()
    save(output / 'result.json', dict(records=records,
        summary={name: stats([row['results'][name]['contour_px']['median'] for row in records]) for name in fits},
        side_definition='Baseline wrist/index-MCP/pinky-MCP plane versus actual camera ray abs cosine<=.35, separated .18s, crop sharpness>=15, skin interior sharpness>=3; common wrist plane excludes forearm',
        motion_blur_can_still_affect_segmentation=True,
        ROI_and_samples_common_to_all_versions=True,
        edge_segmentation_proxy_not_ground_truth=True, RGB_used_only_for_evaluation=True,
        no_new_independent_capture=True))
    print('SIDE_VIEWS', len(records), flush=True)


if __name__ == '__main__':
    main()
