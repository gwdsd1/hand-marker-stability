from common import np, torch, tensor
from smplx import MANO
from smplx.utils import Struct
from smplx.lbs import batch_rodrigues, batch_rigid_transform

ASSET = '/data2/wangzhengwei/hand_joint_identity_20261014/models/mano_right_numeric.npz'
LAYOUT = '/data2/wangzhengwei/hand_joint_identity_20261014/models/initial_marker_layout.npz'
FINGERS = dict(index=[1, 2, 3, 320], middle=[4, 5, 6, 443], pinky=[7, 8, 9, 671], ring=[10, 11, 12, 554], thumb=[13, 14, 15, 744])


class HandModel:
    def __init__(self, offsets, side='left'):
        numeric = {key:value.copy() for key,value in np.load(ASSET).items()}
        if side == 'left':
            mirror = np.array([-1, 1, 1])
            features = np.tile((mirror[:, None] * mirror[None]).reshape(-1), 15)
            for key in ['v_template', 'J']:
                numeric[key] *= mirror
            numeric['shapedirs'] *= mirror[None, :, None]
            numeric['posedirs'] *= mirror[None, :, None] * features[None, None]
            for key in ['hands_mean', 'hands_components']:
                numeric[key] *= np.tile([1, -1, -1], 15)
            numeric['f'] = numeric['f'][:, [0, 2, 1]]
        self.mano = MANO('', data_struct=Struct(**numeric), is_rhand=side == 'right', use_pca=False, flat_hand_mean=True)
        self.faces = numeric['f']
        self.template = tensor(numeric['v_template'])
        self.shape = tensor(numeric['shapedirs'])
        self.pose_dirs = tensor(numeric['posedirs'].reshape(778 * 3, 135).T.reshape(135, 778, 3))
        self.regressor = tensor(numeric['J_regressor'])
        self.parents = self.mano.parents
        self.weights = tensor(numeric['weights'])
        self.pose_standard_deviation = tensor(np.linalg.norm(numeric['hands_components'], axis=1))
        self.pose_basis = tensor(numeric['hands_components'].T) / self.pose_standard_deviation[None]
        self.pose_mean = tensor(numeric['hands_mean'])
        layout = np.load(LAYOUT)
        self.ids, self.names = layout['vertex_ids'], layout['names']
        normals = np.zeros_like(numeric['v_template'])
        vertices = numeric['v_template']
        face_normals = np.cross(vertices[self.faces[:, 1]] - vertices[self.faces[:, 0]], vertices[self.faces[:, 2]] - vertices[self.faces[:, 0]])
        for column in range(3):
            np.add.at(normals, self.faces[:, column], face_normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        self.normals = tensor(normals[self.ids])
        self.anchor = self.template[self.ids] + .004 * self.normals + tensor(offsets)
        self.jbase = self.regressor @ self.template
        self.jshape = torch.einsum('jv,vck->jck', self.regressor, self.shape)
        self.offsets = np.asarray(offsets)
        axes = np.zeros((15, 3))
        for chain in FINGERS.values():
            for ordinal, joint in enumerate(chain[:3]):
                child = self.jbase[chain[ordinal + 1]].numpy() if ordinal < 2 else vertices[chain[-1]]
                axes[joint - 1] = child - self.jbase[joint].numpy()
        axes /= np.maximum(np.linalg.norm(axes, axis=-1, keepdims=True), 1e-12)
        self.axes = tensor(axes)

    def forward(self, beta, pose, orient, world_wrist, attachment=None, vertices=False):
        beta = beta.reshape(1, -1).expand(len(pose), -1) if beta.ndim == 1 else beta
        joints = self.jbase[None] + torch.einsum('bk,jck->bjc', beta, self.jshape)
        rotations = batch_rodrigues(torch.cat([torch.zeros_like(pose[:, :3]), pose], -1).reshape(-1, 3)).reshape(len(pose), 16, 3, 3)
        posed_joints, transforms = batch_rigid_transform(rotations, joints, self.parents)
        feature = (rotations[:, 1:] - torch.eye(3)).reshape(len(pose), 135)
        ids = np.arange(778) if vertices else self.ids
        base = self.template[ids] if vertices else self.anchor
        if not vertices and attachment is not None:
            base = base + attachment
        shaped = base[None] + torch.einsum('bk,vck->bvc', beta, self.shape[ids]) + torch.einsum('bp,pvc->bvc', feature, self.pose_dirs[:, ids])
        skinning = torch.einsum('vj,bjkl->bvkl', self.weights[ids], transforms)
        local = torch.einsum('bvij,bvj->bvi', skinning[:, :, :3, :3], shaped) + skinning[:, :, :3, 3]
        global_rotation = orient if orient.ndim==3 else batch_rodrigues(orient)
        world = torch.einsum('bij,bvj->bvi', global_rotation, local - joints[:, :1]) + world_wrist[:, None]
        joint_world = torch.einsum('bij,bvj->bvi', global_rotation, posed_joints - joints[:, :1]) + world_wrist[:, None]
        return world, joint_world

    def dimensions(self, beta):
        beta = tensor(beta).reshape(-1)
        vertices = self.template + torch.einsum('k,vck->vc', beta, self.shape)
        joints = self.regressor @ vertices
        result = {}
        for name, chain in FINGERS.items():
            points = torch.cat([joints[chain[:3]], vertices[chain[-1]:chain[-1] + 1]], 0)
            for ordinal in range(3):
                result[f'{name}_segment_{ordinal + 1}_mm'] = torch.linalg.norm(points[ordinal + 1] - points[ordinal]) * 1000
            result[name + '_total_mm'] = sum(result[f'{name}_segment_{ordinal + 1}_mm'] for ordinal in range(3))
        result['palm_width_mcp2_to_mcp5_mm'] = torch.linalg.norm(joints[1] - joints[7]) * 1000
        result['palm_length_wrist_to_middle_mcp_mm'] = torch.linalg.norm(joints[0] - joints[4]) * 1000
        result['hand_length_wrist_to_middle_tip_mm'] = torch.linalg.norm(joints[0] - vertices[443]) * 1000
        return result

    def twist(self, pose):
        selected = [joint - 1 for chain in list(FINGERS.values())[:4] for joint in chain[1:3]]
        return (pose.reshape(-1, 15, 3) * self.axes[None]).sum(-1)[:, selected]
