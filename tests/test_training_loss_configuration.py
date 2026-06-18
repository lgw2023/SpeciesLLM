import importlib.util
from pathlib import Path
import unittest


def _load_loss_composition_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "nanoBERT" / "utils" / "loss_composition.py"
    spec = importlib.util.spec_from_file_location("speciesllm_loss_composition", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainingLossConfigurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loss_composition = _load_loss_composition_module()

    def test_combine_pretraining_losses_downweights_gepc_branches(self):
        total = self.loss_composition.combine_pretraining_losses(
            loss_gep=1.0,
            loss_zero_prob=2.0,
            loss_gepc=3.0,
            loss_gepc_zero_prob=4.0,
            gep_weight=1.0,
            zero_prob_weight=1.0,
            gepc_weight=0.1,
            gepc_zero_prob_weight=0.1,
        )

        self.assertAlmostEqual(total, 3.7)

    def test_combine_pretraining_losses_accepts_disabled_optional_branches(self):
        total = self.loss_composition.combine_pretraining_losses(
            loss_gep=2.0,
            loss_zero_prob=None,
            loss_gepc=None,
            loss_gepc_zero_prob=None,
            gep_weight=1.0,
            zero_prob_weight=1.0,
            gepc_weight=0.1,
            gepc_zero_prob_weight=0.1,
        )

        self.assertEqual(total, 2.0)


if __name__ == "__main__":
    unittest.main()
