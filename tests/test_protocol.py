"""Guard invariants that could otherwise invalidate all experiments."""
import unittest
import numpy as np
import torch
from wajepa_robust.adapter import ReliabilityAdapter
from wajepa_robust.corruptions import Fault, apply_fault
from wajepa_robust.data import relative_poses, pose
from wajepa_robust.metrics import displacement, paired_log_bootstrap


class ProtocolTests(unittest.TestCase):
    def test_quaternion_yaw_matches_navsim_with_nonzero_roll_pitch(self):
        from pyquaternion import Quaternion
        q = Quaternion(axis=[1,2,3], angle=.6)
        frame = {"ego2global_translation": [10,20,0], "ego2global_rotation": q.elements.tolist()}
        self.assertAlmostEqual(pose(frame)[2], q.yaw_pitch_roll[0])

    def test_rotation_into_current_ego_frame(self):
        poses = [[1, 0, np.pi/2], [1,1,np.pi/2], [1,2,np.pi/2], [1,3,np.pi/2], [1,4,np.pi/2]]
        np.testing.assert_allclose(relative_poses(poses)[:, :2], [[-3,0],[-2,0],[-1,0],[0,0],[1,0]], atol=1e-6)

    def test_corruptions_are_causal_and_leave_other_views_intact(self):
        x = torch.arange(4.).reshape(1,4,1,1,1,1).expand(1,4,4,3,2,2).clone()
        features = {"history_images": x, "history_trajectory": torch.zeros(1,4,3)}
        y, status = apply_fault(features, Fault("stale",1,2))
        self.assertEqual(y["history_images"][0,:,1,0,0,0].tolist(), [0,0,0,1])
        self.assertEqual(status[0,:,1,1].tolist(), [0,.5,1,1])
        torch.testing.assert_close(y["history_images"][:,:,0], x[:,:,0])
        torch.testing.assert_close(features["history_images"], x)

    def test_clean_adapter_exact_identity_even_after_learning(self):
        adapter = ReliabilityAdapter()
        torch.nn.init.normal_(adapter.up.weight)
        x = torch.randn(1,4096,512)
        status = torch.zeros(1,4,4,2)
        status[...,0] = 1
        torch.testing.assert_close(adapter(x,status), x, rtol=0, atol=0)

    def test_metric_uses_xy_not_heading(self):
        target = np.zeros((8,3)); pred = np.ones((8,3)); pred[:,2] = 100
        self.assertAlmostEqual(displacement(pred,target)["ade_m"], np.sqrt(2))

    def test_paired_comparison_rejects_different_test_scenes(self):
        rows = [{"condition":"base","token":"a"},{"condition":"new","token":"b"}]
        with self.assertRaises(ValueError):
            paired_log_bootstrap(rows,"base","new")


if __name__ == "__main__":
    unittest.main()
