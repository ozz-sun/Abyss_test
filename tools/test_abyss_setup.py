"""tools/abyss_setup.py's Abyss-tools folder: made from ops/abyss-tools with Abyss's path, the scripts follow the
template (a changed one kept as .bak), the README is the user's once written.  The DSpark drafter's step: downloaded
(here a local file), checked and named in the config's env, kept, taken out.  No GPU, no network.

    python -m unittest tools.test_abyss_setup
"""
import hashlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abyss_setup as A  # noqa: E402


class AbyssTools(unittest.TestCase):
    def test_made_from_the_templates(self):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "Abyss-tools"
            A.make_tools(t)
            for name in ("abyss.sh", "common.sh", "README.md"):
                self.assertTrue((t / name).exists(), name)
            self.assertTrue((t / "logs").is_dir())
            common = (t / "common.sh").read_text(encoding="utf-8")
            self.assertIn(f'ABYSS="${{ABYSS_DIR:-{A.ROOT}}}"', common)
            self.assertNotIn("@ABYSS@", common)
            self.assertTrue((t / "abyss.sh").stat().st_mode & 0o111)

    def test_again_updates_the_scripts_and_keeps_the_readme(self):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "Abyss-tools"
            A.make_tools(t)
            (t / "README.md").write_text("my notes\n", encoding="utf-8")
            (t / "abyss.sh").write_text("old\n", encoding="utf-8")
            A.make_tools(t)
            self.assertEqual((t / "README.md").read_text(encoding="utf-8"), "my notes\n")
            self.assertEqual((t / "abyss.sh.bak").read_text(encoding="utf-8"), "old\n")
            self.assertEqual((t / "abyss.sh").read_text(encoding="utf-8"),
                             (A.TOOL_TEMPLATES / "abyss.sh").read_text(encoding="utf-8"))
            self.assertFalse((t / "common.sh.bak").exists())   # unchanged: no backup

    def test_the_tested_quant_is_known_offline(self):
        files = A.KNOWN["UD-Q2_K_XL"]
        self.assertEqual(len(files), 3)
        self.assertTrue(all(len(sha) == 64 for _, _, sha in files))



class DSparkStep(unittest.TestCase):
    """dspark_step against a config in a temporary folder, the drafter 'downloaded' from a local file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.data = d / "Abyss-data"
        self.cfg = d / "abyss-dsv4-test.json"
        self.cfg.write_text(json.dumps({"exe": "strata", "env": {"GPU_MAX_HW_QUEUES": "2"}}), encoding="utf-8")
        src = d / "src" / "drafter.gguf"
        src.parent.mkdir()
        src.write_bytes(b"GGUF drafter" * 100)
        sha = hashlib.sha256(src.read_bytes()).hexdigest()
        self.patches = [mock.patch.object(A, "DSPARK_URL", str(src.parent) + "/{file}"),
                        mock.patch.object(A, "DSPARK_FILE", ("drafter.gguf", src.stat().st_size, sha))]
        for pt in self.patches:
            pt.start()
        self.gpu = [{"index": 0, "vram_gb": 32.0}]
        self.dst = self.data / "models" / A.DSPARK_REPO / "drafter.gguf"

    def tearDown(self):
        for pt in self.patches:
            pt.stop()
        self.tmp.cleanup()

    def run_step(self, fresh=False, had=None, gpus=None, **kw):
        a = Namespace(dspark=kw.get("dspark", False), no_dspark=kw.get("no_dspark", False), yes=True)
        A.dspark_step(a, self.data, self.cfg, fresh, had, gpus or self.gpu, "hip")
        return json.loads(self.cfg.read_text(encoding="utf-8"))["env"]

    def test_a_new_config_asks_and_the_default_is_no(self):
        env = self.run_step(fresh=True)
        self.assertNotIn("STRATA_DS4_DSPARK", env)
        self.assertFalse(self.dst.exists())

    def test_dspark_downloads_checks_and_turns_it_on(self):
        env = self.run_step(dspark=True)
        self.assertEqual(env["STRATA_DS4_DSPARK"], str(self.dst))
        self.assertEqual(env["GPU_MAX_HW_QUEUES"], "2")   # the rest of the env stays
        self.assertTrue(self.dst.exists())
        self.assertTrue(self.dst.with_name("drafter.gguf.sha256").exists())

    def test_a_config_that_has_it_keeps_it_and_no_dspark_takes_it_out(self):
        self.run_step(dspark=True)
        self.assertIn("STRATA_DS4_DSPARK", self.run_step())
        env = self.run_step(no_dspark=True)
        self.assertNotIn("STRATA_DS4_DSPARK", env)
        self.assertTrue(self.dst.exists())   # the file stays

    def test_rewrite_config_keeps_the_drafter_it_had(self):
        self.run_step(dspark=True)
        self.cfg.write_text(json.dumps({"exe": "strata", "env": {}}), encoding="utf-8")   # written again
        env = self.run_step(fresh=True, had=str(self.dst))
        self.assertEqual(env["STRATA_DS4_DSPARK"], str(self.dst))

    def test_not_with_a_layer_split(self):
        env = self.run_step(dspark=True, gpus=self.gpu * 2)
        self.assertNotIn("STRATA_DS4_DSPARK", env)
        self.assertFalse(self.dst.exists())

    def test_a_wrong_file_stops(self):
        with mock.patch.object(A, "DSPARK_FILE", ("drafter.gguf", 1200, "0" * 64)):
            with self.assertRaises(SystemExit):
                self.run_step(dspark=True)
        self.assertFalse(self.dst.with_name("drafter.gguf.sha256").exists())


if __name__ == "__main__":
    unittest.main()
