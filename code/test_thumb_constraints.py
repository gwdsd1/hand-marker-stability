import unittest
from scipy.spatial.transform import Rotation
from common import ROOT, np, torch, tensor
from constrained_thumb import ThumbObservationModel, thumb_jacobian, pseudoinverse, restore_centers


class ThumbConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit = dict(np.load(ROOT / 'baseline/left_fit.npz'))
        cls.model = ThumbObservationModel(cls.fit, preserve_palm_distance=True)
        cls.thumb = tensor(cls.fit['pose'].reshape(-1, 15, 3)[:, 12:].reshape(-1, 9))

    def test_subset_forward_matches_standard_MANO(self):
        constraint, root, unused_rotations, centers = self.model.forward(self.thumb)
        world = torch.einsum('bij,bmj->bmi', self.model.rotation, centers) + self.model.wrist[:, None]
        np.testing.assert_allclose(world.numpy()[:, 2:], self.fit['markers_predicted_world_m'][:, 2:], atol=4e-7, rtol=0)
        expected_local = np.einsum('bji,bvj->bvi', Rotation.from_rotvec(self.fit['orient']).as_matrix(),
            self.fit['vertices_world_m'][:, self.model.root_ids.numpy()] - self.fit['wrist'][:, None])
        np.testing.assert_allclose(root.numpy(), expected_local, atol=4e-7, rtol=0)
        self.assertEqual(constraint.shape[1], 7)

    def test_null_direction_preserves_linearized_constraints(self):
        parameters = self.thumb.clone().requires_grad_(True)
        prediction = self.model.forward(parameters)[0]
        jacobian = thumb_jacobian(prediction, parameters)
        random = torch.randn(parameters.shape, generator=torch.Generator().manual_seed(42))
        direction = random - torch.einsum('bij,bj->bi', pseudoinverse(jacobian), torch.einsum('bij,bj->bi', jacobian, random))
        deviation = torch.einsum('bij,bj->bi', jacobian, direction)
        self.assertLess(float(deviation.norm(dim=-1).max()), 5e-6)

    def test_custom_forward_matches_reference_MANO_skinning(self):
        indices = [0, 224, 412, 430, 633]
        beta = tensor(self.fit['beta'])[None].expand(len(indices), -1)
        pose = tensor(self.fit['pose'][indices])
        orient = tensor(self.fit['orient'][indices])
        root = self.model.joints[0]
        translation = tensor(self.fit['wrist'][indices]) - root
        with torch.no_grad():
            reference = self.model.model.mano(betas=beta, hand_pose=pose, global_orient=orient, transl=translation)
            custom, unused_joints = self.model.model.forward(beta, pose, orient, tensor(self.fit['wrist'][indices]), vertices=True)
        np.testing.assert_allclose(custom.numpy(), reference.vertices.numpy(), atol=4e-7, rtol=0)

    def test_nonlinear_restoration_keeps_marker_centers_and_palm_distance(self):
        parameters = self.thumb.clone().requires_grad_(True)
        target = self.model.forward(parameters)[0]
        jacobian = thumb_jacobian(target, parameters)
        random = torch.randn(parameters.shape, generator=torch.Generator().manual_seed(13))
        direction = random - torch.einsum('bij,bj->bi', pseudoinverse(jacobian), torch.einsum('bij,bj->bi', jacobian, random))
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8) * .002
        restored = restore_centers(self.model, parameters.detach() + direction, target.detach(), iterations=6)
        difference = self.model.forward(restored)[0] - target.detach()
        self.assertLess(float(difference.norm(dim=-1).max()), 2e-7)


if __name__ == '__main__':
    unittest.main()
