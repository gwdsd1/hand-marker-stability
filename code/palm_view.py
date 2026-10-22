from common import np, tensor, torch
from marker_model import MarkerHand


def palm_normals(fit, side='left'):
    model = MarkerHand(np.zeros((14, 3)), side)
    model.set_attachment(fit)
    rest_vertices = model.template + torch.einsum('k,vck->vc', tensor(fit['beta']), model.shape)
    rest_joints = model.regressor @ rest_vertices
    rest_cross = torch.linalg.cross(rest_joints[1] - rest_joints[0], rest_joints[7] - rest_joints[0])
    with torch.no_grad():
        unused, direction = model.surface(rest_vertices[None], fit['marker_vertex_ids'])
    dorsal = direction[0, [7, 9, 11, 13]].mean(0)
    sign = 1. if float(rest_cross @ dorsal) >= 0 else -1.
    joints = fit['joints_world_m']
    cross = np.cross(joints[:, 1] - joints[:, 0], joints[:, 7] - joints[:, 0])
    magnitude = np.linalg.norm(cross, axis=-1, keepdims=True)
    if np.any(magnitude < 1e-9):
        raise ValueError('Degenerate palm MCP plane')
    return sign * cross / magnitude


def facing_camera(normal, wrist, transform):
    local_normal = normal @ transform[:3, :3].T
    local_wrist = wrist @ transform[:3, :3].T + transform[:3, 3]
    towards_camera = -local_wrist / np.linalg.norm(local_wrist)
    return float(local_normal @ towards_camera)
