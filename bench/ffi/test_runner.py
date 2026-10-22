"""Optional runner checks; no Torch import or package/build dependency."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from run import manifest_inputs, parse_args, saved_native_build, sha256


class SavedNativeComparisonTests(unittest.TestCase):
    def test_saved_vm_manifest_requires_vm(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--output", "results", "--before-vm-manifest", "old.json"])

    def test_saved_vm_inputs_detect_header_change(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "adapter.cpp"
            header = Path(directory) / "guard.h"
            source.write_text('#include "guard.h"\n')
            header.write_text("before")
            manifest = Path(directory) / "old.json"
            manifest.write_text(json.dumps(dict(modules=[], native_extensions=[
                dict(sources=[str(source)], dependencies=[str(header)])])))
            before = manifest_inputs(manifest)
            header.write_text("after")
            self.assertNotEqual(before, manifest_inputs(manifest))

    def test_requires_matching_build_log(self):
        for options in (("--before-native", "old"), ("--before-build-log", "old.log")):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--output", "results", *options])

    def test_saved_native_can_be_combined_with_saved_vm(self):
        args = parse_args(["--output", "results", "--before-native", "old",
                           "--before-build-log", "old.log", "--before-vm", "vm"])
        self.assertEqual(args.before_native, Path("old"))
        self.assertEqual(args.before_vm, Path("vm"))

    def test_rejects_stale_binary_and_partial_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory) / "native"
            native.write_bytes(b"saved native executable")
            log = Path(directory) / "build.log"
            build = dict(native_graph_full_coverage=True, native_bytecode_fallback=False,
                         native_hash=sha256(native))
            log.write_text("build progress\n" + json.dumps(build) + "\nfinished\n")
            self.assertEqual(saved_native_build(native, log), build)
            for key, value in (("native_graph_full_coverage", False),
                               ("native_bytecode_fallback", True),
                               ("native_hash", "stale")):
                with self.subTest(key=key):
                    log.write_text(json.dumps({**build, key: value}))
                    with self.assertRaisesRegex(ValueError, "binary/coverage"):
                        saved_native_build(native, log)
            log.write_text(json.dumps(build))
            native.write_bytes(b"changed executable")
            with self.assertRaisesRegex(ValueError, "binary/coverage"):
                saved_native_build(native, log)


if __name__ == "__main__":
    unittest.main()
