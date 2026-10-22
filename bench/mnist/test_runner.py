"""Runner acceptance checks; optional benchmark tooling, not tools/test."""

import argparse
import copy
import contextlib
import io
import json
import unittest

import run
import compare_calls


def sample(lane="python"):
    return dict(schema="torch.mnist.sample.v1", lane=lane, device="cpu", version="2.14.0",
                threads=4, interop_threads=10, train_count=64, test_count=32, parameters=19706,
                execution_scope=None if lane == "python" else "run",
                batch_size=16, seed=7, optimizer="matched", profiled=False, data_seconds=0.1,
                setup_seconds=0.1, save_seconds=0.1, restore_seconds=0.1, restore_eval_seconds=0.1,
                workload_seconds=2.0, checkpoint_loss=0.5, checkpoint_accuracy=0.75,
                epochs=[dict(epoch=1, train_loss=1.0, test_loss=0.5, accuracy=0.75,
                             train_seconds=1.0, eval_seconds=0.2, profile=None)])


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(train_limit=64, test_limit=32, seed=7, batch_size=16,
                                       device="cpu", profile=False, epochs=1, threads=4,
                                       execution_scope="run",
                                       loss_atol=1e-5, accuracy_atol=0)

    def test_valid_metrics_and_timing(self):
        parsed = run.parse_sample(run.PREFIX + json.dumps(sample()), 2.5)
        self.assertAlmostEqual(parsed["loop_seconds"], 1.2)
        self.assertAlmostEqual(parsed["startup_teardown_seconds"], 0.5)
        run.validate_samples([parsed], self.args)

    def test_rejects_duplicate_summary(self):
        line = run.PREFIX + json.dumps(sample())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            run.parse_sample(line + "\n" + line, 2.5)

    def test_rejects_checkpoint_drift(self):
        value = sample()
        value["checkpoint_loss"] = 0.6
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            run.parse_sample(run.PREFIX + json.dumps(value), 2.5)

    def test_rejects_nonfinite_loss(self):
        value = sample()
        value["epochs"][0]["train_loss"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            run.parse_sample(run.PREFIX + json.dumps(value), 2.5)

    def test_rejects_different_workload(self):
        value = sample("sputnik-vm")
        value["train_count"] = 63
        with self.assertRaisesRegex(ValueError, "split sizes"):
            run.validate_samples([sample(), value], self.args)

    def test_rejects_numerical_difference(self):
        value = sample("sputnik-built")
        value["epochs"][0]["train_loss"] = 1.1
        with self.assertRaisesRegex(ValueError, "Numerical mismatch"):
            run.validate_samples([sample(), value], self.args)

    def test_rejects_libtorch_version_difference(self):
        value = sample("sputnik-built")
        value["version"] = "2.14.1"
        with self.assertRaisesRegex(ValueError, "versions"):
            run.validate_samples([sample(), value], self.args)

    def test_rejects_different_execution_scope(self):
        value = sample("sputnik-built")
        value["execution_scope"] = "epoch"
        with self.assertRaisesRegex(ValueError, "execution scope"):
            run.validate_samples([sample(), value], self.args)

    def test_baseline_regression_direction(self):
        baseline = dict(config=dict(device="cpu", epochs=1, batch_size=16, seed=7,
                                    train_limit=64, test_limit=32, threads=4, profile=False,
                                    python_optimizer="matched"),
                        host=dict(machine="arm64", processor="test", torch_version="2.14.0"),
                        summary={lane: dict(loop_seconds=dict(median=1.0),
                                           train_images_per_second=dict(median=100.0))
                                 for lane in run.LANES})
        current = copy.deepcopy(baseline)
        baseline["config"]["execution_scope"] = "run"
        current["config"]["execution_scope"] = "run"
        current["summary"]["sputnik-vm"]["loop_seconds"]["median"] = 1.3
        current["summary"]["sputnik-built"]["train_images_per_second"]["median"] = 70
        changes = run.compare_baseline(current, baseline, 0.15)
        self.assertEqual(sum(c["regression"] for c in changes), 2)
        current["config"]["profile"] = True
        with self.assertRaisesRegex(ValueError, "different workload"):
            run.compare_baseline(current, baseline, 0.15)

    def test_speed_check_rejects_one_slow_pair_despite_faster_median(self):
        values = []
        for repeat, seconds in enumerate((0.8, 0.8, 1.01)):
            for lane in run.LANES:
                values.append(dict(lane=lane, repeat=repeat, warmup=False,
                                   loop_seconds=seconds if lane == "sputnik-vm" else
                                   0.8 if lane == "sputnik-built" else 1.0))
        values.append(dict(lane="sputnik-vm", repeat=0, warmup=True, loop_seconds=100.0))
        check = run.check_python_speed(values, 1.0)
        self.assertFalse(check["passed"])
        self.assertEqual(sum(not row["passed"] for row in check["comparisons"]), 1)

    def test_speed_check_requires_the_requested_margin_and_rejects_ties(self):
        values = [dict(lane=lane, repeat=0, warmup=False,
                       loop_seconds=0.98 if lane != "python" else 1.0)
                  for lane in run.LANES]
        self.assertTrue(run.check_python_speed(values, 1.0)["passed"])
        self.assertFalse(run.check_python_speed(values, 1.05)["passed"])
        values[0]["loop_seconds"] = 1.0
        self.assertFalse(run.check_python_speed(values, 1.0)["passed"])


class CompareCallsTests(unittest.TestCase):
    common = ["--device", "cpu", "--python", "/optional/python", "--output", "/tmp/ab"]

    def test_vm_only_comparison_accepts_batch_workload(self):
        args = compare_calls.parse_args(self.common + ["--before-vm", "/tmp/vm", "--batch-size", "128"])
        self.assertIsNone(args.before_native)
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.execution_scope, "run")

    def test_saved_native_requires_its_matching_provenance_argument(self):
        for extra in (["--before-native", "/tmp/native"],
                      ["--before-vm", "/tmp/vm", "--before-build-log", "/tmp/log"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    compare_calls.parse_args(self.common + extra)
                self.assertEqual(error.exception.code, 2)

    def test_rejects_missing_baseline_and_invalid_batch(self):
        for extra in ([], ["--before-vm", "/tmp/vm", "--batch-size", "0"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    compare_calls.parse_args(self.common + extra)
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
