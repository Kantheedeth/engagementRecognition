import ast
import json
from pathlib import Path
import unittest

from experiments_v2.certification.environment import inspect_environment
from experiments_v2.certification.preflight import build_preflight_report
from experiments_v2.core.config import load_config
from experiments_v2.registry.builtins import create_builtin_registry


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def argparse_defaults(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        default = next(
            (item.value for item in node.keywords if item.arg == "default"), None
        )
        if default is not None:
            try:
                defaults[flag[2:]] = ast.literal_eval(default)
            except (ValueError, TypeError):
                continue
    return defaults


class BaselineEquivalenceTests(unittest.TestCase):
    def test_v2_baseline_matches_legacy_training_defaults(self):
        config = json.loads(
            (PROJECT_ROOT / "experiments_v2/config/baseline_legacy.json").read_text(
                encoding="utf-8"
            )
        )
        legacy = argparse_defaults(PROJECT_ROOT / "src/training/train_behavioral.py")
        model = config["engagement"]["model"]
        training = config["engagement"]["training"]
        evaluation = config["engagement"]["evaluation"]
        registry = create_builtin_registry()

        self.assertEqual(registry.spec("I1").feature_dim, legacy["dim_inter"])
        self.assertEqual(registry.spec("A1").feature_dim, legacy["dim_affect"])
        self.assertEqual(model["branch_dim"], legacy["branch_dim"])
        self.assertEqual(model["num_heads"], legacy["num_heads"])
        self.assertEqual(model["dropout"], legacy["dropout"])
        self.assertEqual(training["seed"], legacy["seed"])
        self.assertEqual(training["epochs"], legacy["epochs"])
        self.assertEqual(training["batch_size"], legacy["batch_size"])
        self.assertEqual(training["learning_rate"], legacy["lr"])
        self.assertEqual(training["weight_decay"], legacy["weight_decay"])
        self.assertEqual(training["patience"], legacy["patience"])
        self.assertEqual(evaluation["batch_size"], legacy["batch_size"])
        self.assertEqual(config["pairing"]["feature_order"], ["interaction", "affect"])

    def test_declared_split_sizes_match_legacy_baseline_report(self):
        expected = {"train": 939, "val": 124, "test": 132}
        for split, expected_count in expected.items():
            records = [
                line
                for line in (PROJECT_ROOT / f"{split}.csv").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), expected_count)

        config = load_config(
            PROJECT_ROOT / "experiments_v2/config/baseline_legacy.json",
            PROJECT_ROOT,
        )
        environment = inspect_environment(config["certification"]["required_python"])
        self.assertIn(environment["status"], {"READY", "NOT_READY"})
        self.assertIn("reuse_and_training", environment["profiles"])
        self.assertIn("feature_extraction_and_training", environment["profiles"])
        preflight = build_preflight_report(config)
        self.assertIn(preflight["status"], {"READY", "NOT_READY"})
        self.assertIn("reuse_legacy_artifacts", preflight["routes"])
        self.assertIn("rebuild_from_preprocessed", preflight["routes"])


if __name__ == "__main__":
    unittest.main()
