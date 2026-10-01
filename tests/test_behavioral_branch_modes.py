"""Behavioral branch-ablation model tests."""

import unittest

import torch

from src.models.model_behavioral import PureBehavioralAttentionClassifier


class BehavioralBranchModeTests(unittest.TestCase):
    def test_all_branch_modes_produce_three_class_logits(self):
        x = torch.randn(3, 8, 40)
        for mode in ("both", "interaction", "affect"):
            model = PureBehavioralAttentionClassifier(
                branch_dim=16, num_heads=4, branch_mode=mode
            )
            self.assertEqual(tuple(model(x).shape), (3, 3))

    def test_invalid_branch_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "branch_mode"):
            PureBehavioralAttentionClassifier(branch_mode="invalid")

    def test_selected_interaction_columns_produce_logits(self):
        kept = [index for index in range(32) if index not in (2, 3, 4, 5, 6, 7, 30)]
        model = PureBehavioralAttentionClassifier(
            branch_dim=16, num_heads=4, interaction_indices=kept
        )
        self.assertEqual(model.branch_inter[0].in_features, 25)
        self.assertEqual(tuple(model(torch.randn(3, 8, 40)).shape), (3, 3))

    def test_dropped_interaction_columns_cannot_change_logits(self):
        dropped = (2, 3, 4, 5, 6, 7, 30)
        kept = [index for index in range(32) if index not in dropped]
        model = PureBehavioralAttentionClassifier(
            branch_dim=16, num_heads=4, dropout=0.0,
            branch_mode="both", interaction_indices=kept,
        ).eval()
        original = torch.randn(3, 8, 40)
        changed = original.clone()
        changed[:, :, list(dropped)] += 1000.0
        with torch.no_grad():
            torch.testing.assert_close(model(original), model(changed))

    def test_invalid_interaction_indices_are_rejected(self):
        for indices in ([], [0, 0], [-1, 1], [0, 32]):
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(ValueError, "interaction_indices"):
                    PureBehavioralAttentionClassifier(interaction_indices=indices)


if __name__ == "__main__":
    unittest.main()
