from common import np
import cv2


def project(points, projection):
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    image = homogeneous @ projection.T
    return image[:, :2] / image[:, 2:3]


def rasterize(vertices, faces, projection, origin, shape):
    projected = project(vertices, projection) - origin
    depth = np.full(tuple(shape), np.inf, np.float32)
    for triangle in faces:
        xy = projected[triangle]
        camera_depth = vertices[triangle, 2]
        if np.min(camera_depth) <= .05:
            continue
        lower = np.maximum(np.floor(xy.min(0)).astype(int), 0)
        upper = np.minimum(np.ceil(xy.max(0)).astype(int), np.array(shape)[::-1] - 1)
        if (upper < lower).any():
            continue
        columns, rows = np.meshgrid(np.arange(lower[0], upper[0] + 1), np.arange(lower[1], upper[1] + 1))
        divisor = (xy[1, 1] - xy[2, 1]) * (xy[0, 0] - xy[2, 0]) + (xy[2, 0] - xy[1, 0]) * (xy[0, 1] - xy[2, 1])
        if abs(divisor) < 1e-8:
            continue
        first = ((xy[1, 1] - xy[2, 1]) * (columns - xy[2, 0]) + (xy[2, 0] - xy[1, 0]) * (rows - xy[2, 1])) / divisor
        second = ((xy[2, 1] - xy[0, 1]) * (columns - xy[2, 0]) + (xy[0, 0] - xy[2, 0]) * (rows - xy[2, 1])) / divisor
        third = 1 - first - second
        inside = (first >= -1e-6) & (second >= -1e-6) & (third >= -1e-6)
        inverse = first / camera_depth[0] + second / camera_depth[1] + third / camera_depth[2]
        candidate = np.where(inside, 1 / np.maximum(inverse, 1e-6), np.inf)
        region = depth[lower[1]:upper[1] + 1, lower[0]:upper[0] + 1]
        np.minimum(region, candidate, out=region)
    return depth, projected


def score_render(depth, observation):
    columns, rows = observation['uv'].T
    rendered = depth[rows, columns]
    missing = ~np.isfinite(rendered)
    errors = np.minimum(np.abs(rendered - observation['depth']) * 1000, 30.)
    errors[missing] = 30.
    shape = tuple(observation['shape'])
    truth = np.unpackbits(observation['segmentation'])[:np.prod(shape)].reshape(shape).astype(np.uint8)
    predicted = np.isfinite(depth).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    actual_edge = truth - cv2.erode(truth, kernel)
    predicted_edge = predicted - cv2.erode(predicted, kernel)
    actual_edge[[0, -1], :] = 0
    actual_edge[:, [0, -1]] = 0
    predicted_edge[[0, -1], :] = 0
    predicted_edge[:, [0, -1]] = 0
    if actual_edge.any() and predicted_edge.any():
        distance_actual = cv2.distanceTransform(1 - actual_edge, cv2.DIST_L2, 5)
        distance_predicted = cv2.distanceTransform(1 - predicted_edge, cv2.DIST_L2, 5)
        boundary = np.concatenate([distance_actual[predicted_edge > 0], distance_predicted[actual_edge > 0]])
        edge_score = float(np.quantile(boundary, .75))
    else:
        edge_score = 100.
    q75 = float(np.quantile(errors, .75))
    reverse_q75 = q75
    reverse_unknown = 0.
    reverse_points = 0
    if 'scene_depth' in observation:
        scene = observation['scene_depth']
        excluded = np.unpackbits(observation['marker_exclusion'])[:np.prod(shape)].reshape(shape).astype(bool)
        rows_grid, columns_grid = np.indices(shape)
        grid = ((rows_grid + observation.get('origin', [0, 0])[1]) % 3 == 0)
        grid &= ((columns_grid + observation.get('origin', [0, 0])[0]) % 3 == 0)
        rendered_support = grid & np.isfinite(depth) & ~excluded
        reliable = rendered_support & (scene > 0)
        reverse_points = int(reliable.sum())
        reverse_unknown = float(1 - reliable.sum() / max(rendered_support.sum(), 1))
        if reverse_points:
            reverse_q75 = float(np.quantile(np.minimum(np.abs(depth[reliable] - scene[reliable]) * 1000, 30.), .75))
        else:
            reverse_q75 = 30.
    symmetric = .5 * (q75 + reverse_q75)
    return dict(depth_q75_mm=q75, depth_median_mm=float(np.median(errors)),
        depth_mean_mm=float(np.mean(errors)), missing_fraction=float(missing.mean()),
        reverse_depth_q75_mm=reverse_q75, reverse_unknown_fraction=reverse_unknown, reverse_points=reverse_points,
        symmetric_depth_q75_mm=symmetric,
        silhouette_boundary_q75_px=edge_score, silhouette_iou=float((truth & predicted).sum() / max((truth | predicted).sum(), 1)),
        combined_score=float(symmetric + 10 * missing.mean() + 3 * reverse_unknown), points=len(errors))


def overlay(image, depth, color):
    result = image.copy()
    mask = np.isfinite(depth)
    result[mask] = (.7 * result[mask] + .3 * np.asarray(color)).astype(np.uint8)
    edge = mask.astype(np.uint8) - cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    result[edge > 0] = color
    return result
