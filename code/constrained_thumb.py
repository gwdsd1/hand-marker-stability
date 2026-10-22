import argparse
import shutil
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rigid_transform, batch_rodrigues
from common import ROOT, np, torch, tensor, assert_open
from marker_model import MarkerHand
from io_utils import save


class ThumbObservationModel:
    def __init__(self, fit, preserve_palm_distance=False, constraint_kind='centers'):
        self.fit = fit
        self.model = MarkerHand(np.zeros((14, 3)), 'left')
        self.model.set_attachment(fit)
        self.model.set_shape_axes(fit['beta'])
        if np.max(abs(fit['marker_height_m'][2:])) > 1e-10:
            raise ValueError('This exact-center constrained solver requires the inherited zero-height hypothesis')
        self.base_pose = tensor(fit['pose'].reshape(-1, 15, 3))
        self.rotation = tensor(Rotation.from_rotvec(fit['orient']).as_matrix())
        self.wrist = tensor(fit['wrist'])
        self.shaped = self.model.template + torch.einsum('k,vck->vc', tensor(fit['beta']), self.model.shape)
        self.joints = self.model.regressor @ self.shaped
        distance = (self.shaped - self.joints[:1]).norm(dim=-1)
        self.root_ids = torch.nonzero((self.model.weights[:, 0] > .9) & (distance < .045), as_tuple=False).flatten()
        self.selected = torch.unique(torch.cat([tensor(fit['marker_attachment_faces']).long().reshape(-1), self.root_ids]))
        lookup = torch.full((778,), -1, dtype=torch.long)
        lookup[self.selected] = torch.arange(len(self.selected))
        self.face_index = lookup[tensor(fit['marker_attachment_faces']).long()]
        self.root_index = lookup[self.root_ids]
        self.barycentric = tensor(fit['marker_attachment_barycentric'])
        self.preserve_palm_distance = preserve_palm_distance
        self.constraint_kind = constraint_kind
        observation = dict(np.load(ROOT / 'inputs_v2/left.npz'))
        self.palm_observed = torch.einsum('bji,bj->bi', self.rotation,
            tensor(observation['markers_world_m'][:, 2]) - self.wrist)
        self.observed_local = torch.einsum('bji,bmj->bmi', self.rotation,
            tensor(np.nan_to_num(observation['markers_world_m'])) - self.wrist[:, None])

    def forward(self, thumb):
        count = len(thumb)
        pose = torch.cat([self.base_pose[:, :12], thumb.reshape(count, 3, 3)], 1)
        rotations = batch_rodrigues(torch.cat([torch.zeros(count, 1, 3), pose], 1).reshape(-1, 3)).reshape(count, 16, 3, 3)
        unused_joints, transforms = batch_rigid_transform(rotations, self.joints[None].expand(count, -1, -1), self.model.parents)
        self.current_joints = unused_joints - self.joints[:1]
        feature = (rotations[:, 1:] - torch.eye(3)).reshape(count, 135)
        shaped = self.shaped[self.selected][None] + torch.einsum('bp,pvc->bvc', feature, self.model.pose_dirs[:, self.selected])
        skinning = torch.einsum('vj,bjkl->bvkl', self.model.weights[self.selected], transforms)
        vertices = torch.einsum('bvij,bvj->bvi', skinning[:, :, :3, :3], shaped) + skinning[:, :, :3, 3] - self.joints[:1]
        centers = (vertices[:, self.face_index] * self.barycentric[None, :, :, None]).sum(2)
        constraint = (centers[:, 4:6] - self.observed_local[:, 4:6]).norm(dim=-1) if self.constraint_kind == 'residual-norms' else centers[:, 4:6].reshape(count, 6)
        if self.preserve_palm_distance:
            distance = (centers[:, 2] - self.palm_observed).norm(dim=-1, keepdim=True)
            constraint = torch.cat([constraint, distance], -1)
        return constraint, vertices[:, self.root_index], rotations[:, 13:], centers


def thumb_jacobian(prediction, parameters):
    coordinates = prediction.reshape(len(parameters), -1)
    return torch.stack([torch.autograd.grad(coordinates[:, column].sum(), parameters,
        retain_graph=True)[0] for column in range(coordinates.shape[1])], 1)


def pseudoinverse(jacobian):
    return torch.linalg.pinv(jacobian, rtol=1e-5)


def restore_centers(model, value, target, iterations=5):
    current = value.detach()
    for unused in range(iterations):
        current = current.requires_grad_(True)
        prediction, unused_root, unused_rotations, unused_centers = model.forward(current)
        residual = (target - prediction).reshape(len(current), -1)
        jacobian = thumb_jacobian(prediction, current)
        delta = torch.einsum('bij,bj->bi', pseudoinverse(jacobian), residual.detach())
        delta = delta * (.15 / delta.norm(dim=-1, keepdim=True).clamp_min(.15))
        current = (current + delta).detach()
    return current


def objective(model, value, times, original, palm_weight, twist_weight, loss_type='huber', joint_weight=0, maximum_step_weight=0, minimum_axial=False):
    prediction, root, rotations, unused_centers = model.forward(value)
    intervals = tensor(np.diff(times))
    continuous = tensor((np.diff(times)[1:] < .12) & (np.diff(times)[:-1] < .12)).bool()
    rotation_velocity = (rotations[1:] - rotations[:-1]) / intervals[:, None, None, None]
    rotation_acceleration = 2 * (rotation_velocity[1:] - rotation_velocity[:-1]) / (intervals[1:] + intervals[:-1])[:, None, None, None]
    angular_square = rotation_acceleration[continuous].square().sum((-1, -2)) / 100
    angular = (torch.sqrt(1 + angular_square) - 1).mean() if loss_type == 'huber' else torch.log1p(angular_square).mean()
    root_velocity = (root[1:] - root[:-1]) / intervals[:, None, None]
    root_acceleration = 2 * (root_velocity[1:] - root_velocity[:-1]) / (intervals[1:] + intervals[:-1])[:, None, None]
    palm = torch.log1p(root_acceleration[continuous].square().sum(-1) / .01).mean()
    axes = model.model.axes[12:]
    angle = value.reshape(-1, 3, 3).norm(dim=-1).clamp_min(1e-8)
    axial = (value.reshape(-1, 3, 3) * axes[None]).sum(-1) / angle
    twist = 2 * torch.atan2(torch.sin(angle / 2) * axial, torch.cos(angle / 2))
    twist_prior = torch.relu(twist[:, 1:].abs() - .35).square().mean()
    if minimum_axial:
        twist_prior = twist[:, 1:].square().mean()
    prior = ((value - original) / .5).square().mean()
    world_joints = torch.einsum('bij,bvj->bvi', model.rotation, model.current_joints[:, 13:]) + model.wrist[:, None]
    joint_velocity = (world_joints[1:] - world_joints[:-1]) / intervals[:, None, None]
    joint_acceleration = 2 * (joint_velocity[1:] - joint_velocity[:-1]) / (intervals[1:] + intervals[:-1])[:, None, None]
    joint = (torch.sqrt(1 + joint_acceleration[continuous].square().sum(-1) / 25) - 1).mean()
    pair_mask = tensor(np.diff(times) < .12).bool()
    relative = rotations[:-1].transpose(-1, -2) @ rotations[1:]
    chordal = ((relative[pair_mask] - torch.eye(3)).square().sum((-1, -2)) / 8).reshape(-1)
    maximum_step = (torch.logsumexp(chordal * 20, 0) - np.log(len(chordal))) / 20
    loss = angular + palm_weight * palm + twist_weight * twist_prior + .002 * prior + joint_weight * joint + maximum_step_weight * maximum_step
    return loss, prediction, dict(angular=float(angular.detach()), palm=float(palm.detach()), twist=float(twist_prior.detach()), joint=float(joint.detach()), maximum_step=float(maximum_step.detach()))


def save_fit(model, thumb, output):
    source = model.fit
    result = {name:value.copy() for name,value in source.items()}
    result['pose'] = source['pose'].copy().reshape(-1, 15, 3)
    result['pose'][:, 12:] = thumb.numpy().reshape(-1, 3, 3)
    result['pose'] = result['pose'].reshape(-1, 45)
    with torch.no_grad():
        vertices, joints, prediction = model.model.predict(tensor(source['beta']), tensor(result['pose']),
            tensor(source['orient']), tensor(source['wrist']), source['marker_vertex_ids'],
            tensor(source['marker_height_m']), tensor(source['wrist_marker_local_offsets_m']))
        if 'wrist_articulation_matrix' in source:
            local = torch.einsum('bij,mj->bmi', tensor(source['wrist_articulation_matrix']), tensor(source['wrist_marker_local_offsets_m']))
            prediction[:, :2] = torch.einsum('bij,bmj->bmi', model.rotation, local) + model.wrist[:, None]
    result.update(vertices_world_m=vertices.numpy(), joints_world_m=joints.numpy(), markers_predicted_world_m=prediction.numpy())
    np.savez_compressed(output / 'left_fit.npz', **result)
    shutil.copy2(ROOT / 'baseline/right_fit.npz', output / 'right_fit.npz')
    return result


def main():
    assert_open()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--step-rad', type=float, default=.025)
    parser.add_argument('--palm-weight', type=float, default=1)
    parser.add_argument('--twist-weight', type=float, default=.3)
    parser.add_argument('--initial-fit', default='baseline')
    parser.add_argument('--preserve-palm-distance', action='store_true')
    parser.add_argument('--objective-loss', choices=['huber', 'log'], default='huber')
    parser.add_argument('--active-mask')
    parser.add_argument('--active-mask-field', default='thumb_ambiguity_active')
    parser.add_argument('--constraint-kind', choices=['centers', 'residual-norms'], default='centers')
    parser.add_argument('--joint-weight', type=float, default=0)
    parser.add_argument('--maximum-step-weight', type=float, default=0)
    parser.add_argument('--minimum-axial-compensation', action='store_true')
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(exist_ok=False)
    source = dict(np.load(ROOT / 'baseline/left_fit.npz'))
    model = ThumbObservationModel(source, args.preserve_palm_distance, args.constraint_kind)
    original = tensor(source['pose'].reshape(-1, 15, 3)[:, 12:].reshape(-1, 9))
    initial = dict(np.load(ROOT / args.initial_fit / 'left_fit.npz'))
    current = tensor(initial['pose'].reshape(-1, 15, 3)[:, 12:].reshape(-1, 9))
    active = tensor(np.load(ROOT / args.active_mask)[args.active_mask_field]).bool() if args.active_mask else torch.ones(len(current), dtype=torch.bool)
    current[~active] = original[~active]
    target = model.forward(original)[0].detach()
    history = []
    times = source['timestamps_seconds']
    for step in range(args.steps):
        current = current.detach().requires_grad_(True)
        loss, prediction, components = objective(model, current, times, original, args.palm_weight, args.twist_weight, args.objective_loss, args.joint_weight, args.maximum_step_weight, args.minimum_axial_compensation)
        gradient = torch.autograd.grad(loss, current, retain_graph=True)[0]
        jacobian = thumb_jacobian(prediction, current).detach()
        inverse = pseudoinverse(jacobian)
        projected = gradient - torch.einsum('bij,bj->bi', inverse, torch.einsum('bij,bj->bi', jacobian, gradient))
        scale = torch.quantile(projected[active].norm(dim=-1), .75).clamp_min(1e-8)
        delta = -projected / scale * args.step_rad
        delta *= (args.step_rad * 3 / delta.norm(dim=-1, keepdim=True).clamp_min(args.step_rad * 3))
        delta[~active] = 0
        accepted = False
        for fraction in [2. ** -index for index in range(9)]:
            proposal = restore_centers(model, current.detach() + fraction * delta, target)
            proposal[~active] = original[~active]
            with torch.no_grad():
                candidate_loss, candidate_prediction, candidate_components = objective(model, proposal, times, original, args.palm_weight, args.twist_weight, args.objective_loss, args.joint_weight, args.maximum_step_weight, args.minimum_axial_compensation)
                error = (candidate_prediction - target).norm(dim=-1).max()
            if candidate_loss < loss.detach() and error < 2e-7:
                current = proposal
                accepted = True
                break
        if not accepted:
            current = current.detach()
        record = dict(step=step, before_loss=float(loss.detach()), after_loss=float(candidate_loss),
            accepted=accepted, maximum_center_displacement_mm=float(error * 1000), **components)
        history.append(record)
        if step % 10 == 0 or not accepted:
            print(record, flush=True)
        if not accepted:
            break
    result = save_fit(model, current.detach(), output)
    displacement = np.linalg.norm(result['markers_predicted_world_m'] - source['markers_predicted_world_m'], axis=-1) * 1000
    save(output / 'result.json', dict(history=history, options=vars(args),
        thumb_center_max_displacement_mm=float(displacement[:, 4:6].max()),
        other_center_max_displacement_mm=float(displacement[:, np.r_[0:4, 6:14]].max()),
        fixed_other_finger_pose=np.array_equal(source['pose'][:, :36], result['pose'][:, :36]),
        fixed_root=True, fixed_shared_shape_installation_height=True,
        pose_inputs=['C3D-derived baseline marker centers', 'MANO model', 'time and minimum-compensation priors'],
        RGB_pose_loss=False, depth_used=False, marker_equivalence_not_independent_pose_truth=True))


if __name__ == '__main__':
    main()
