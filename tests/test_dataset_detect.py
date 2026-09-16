"""Recognising a dataset that somebody else laid out.

The app used to accept exactly one shape — ``images/train`` + ``labels/train`` + a ``*.yaml`` —
and silently skipped everything else, including the sample dataset it ships in ``data/``. These
tests pin the layouts and class-name sources that must keep working, because the failure mode is
invisible: the dataset simply does not appear on the start screen and the user assumes the folder
is wrong.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from support import ROOT                                            # noqa: E402
from pglabel import dataset_detect as dd                            # noqa: E402
from pglabel.labelio import is_image                                # noqa: E402


def _img(p: Path, n=2, ext=".jpg"):
    from PIL import Image
    p.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (32, 24)).save(p / f"im{i}{ext}")


def _lab(p: Path, n=2, n_cls=2):
    p.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (p / f"im{i}.txt").write_text(f"{i % n_cls} 0.5 0.5 0.2 0.2\n")


class Layouts(unittest.TestCase):
    """Every layout people actually hand the tool has to resolve to the same answer."""

    def _case(self, build, want_layout, want_labels=True):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            build(root)
            got = dd.resolve_layout(root)
            self.assertIsNotNone(got, f"{want_layout} was not recognised at all")
            self.assertIn(want_layout.split(" ")[0], got["layout"])
            self.assertEqual(got["n_images"], 2)
            self.assertEqual(bool(got["labels"]), want_labels)

    def test_ultralytics_split(self):
        self._case(lambda r: (_img(r / "images" / "train"), _lab(r / "labels" / "train")),
                   "images/train")

    def test_flat(self):
        self._case(lambda r: (_img(r / "images"), _lab(r / "labels")), "images")

    def test_roboflow(self):
        self._case(lambda r: (_img(r / "train" / "images"), _lab(r / "train" / "labels")),
                   "train/images")

    def test_images_only_is_a_dataset(self):
        """An unlabeled folder is the NORMAL start for an auto-labeling tool, not an error."""
        self._case(lambda r: _img(r / "images"), "images", want_labels=False)

    def test_split_wins_over_flat_when_both_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            _img(root / "images" / "train"), _lab(root / "labels" / "train")
            _img(root / "images", 1)
            self.assertIn("images/train", dd.resolve_layout(root)["layout"])

    def test_folder_without_images_is_not_a_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "empty"
            root.mkdir()
            self.assertIsNone(dd.resolve_layout(root))


class ClassNames(unittest.TestCase):
    """All three data.yaml spellings, the darknet file, and the inference fallback."""

    def _names(self, text=None, cls_file=None, labels=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            _img(root / "images", n=3)
            if labels:
                _lab(root / "labels", n=3, n_cls=3)      # ids 0,1,2 all actually appear
            if text is not None:
                (root / "data.yaml").write_text(text)
            if cls_file is not None:
                (root / "classes.txt").write_text(cls_file)
            lay = dd.resolve_layout(root)
            return dd.resolve_class_names(root, lay["labels"] if lay else None)

    def test_yaml_inline_list(self):
        self.assertEqual(self._names("names: [cat, dog]\n")[0], ["cat", "dog"])

    def test_yaml_index_map(self):
        self.assertEqual(self._names("names:\n  0: cat\n  1: dog\nnc: 2\n")[0], ["cat", "dog"])

    def test_yaml_dash_list(self):
        """YOLOv5-era exports use this and it used to parse as zero classes."""
        self.assertEqual(self._names("nc: 2\nnames:\n  - cat\n  - dog\n")[0], ["cat", "dog"])

    def test_classes_txt(self):
        self.assertEqual(self._names(cls_file="cat\ndog\n")[0], ["cat", "dog"])

    def test_yaml_beats_classes_txt(self):
        names, src = self._names("names: [a, b]\n", cls_file="cat\ndog\n")
        self.assertEqual((names, src), (["a", "b"], "data.yaml"))

    def test_inferred_from_label_ids(self):
        names, src = self._names(labels=True)
        self.assertEqual(src, "labels")
        self.assertEqual(names, ["class0", "class1", "class2"])   # sized by the ids actually used

    def test_nothing_to_go_on(self):
        self.assertEqual(self._names()[0], [])


class Extensions(unittest.TestCase):
    """The app and the training loader must agree on what counts as an image."""

    def test_case_insensitive_and_modern_formats(self):
        for name in ("a.JPG", "b.Jpeg", "c.webp", "d.TIF", "e.tiff", "f.PNG"):
            self.assertTrue(is_image(Path(name)), name)
        for name in ("g.txt", "h.yaml", "i.json"):
            self.assertFalse(is_image(Path(name)), name)

    def test_app_and_trainer_agree(self):
        from tools.common import IMG_EXTS as TOOL_EXTS
        from pglabel.labelio import IMG_EXTS as APP_EXTS
        self.assertEqual(set(APP_EXTS), set(TOOL_EXTS))


class BundledSample(unittest.TestCase):
    """The dataset this repo ships must be discoverable by the app that ships it."""

    def test_data_folder_resolves(self):
        info = dd.describe(ROOT / "data")
        self.assertIsNotNone(info, "the bundled data/ sample is not recognised as a dataset")
        self.assertEqual(info["class_source"], "classes.txt")
        self.assertEqual(info["classes"], ["buffalo", "elephant", "rhino", "zebra"])
        self.assertGreater(info["n_images"], 0)


if __name__ == "__main__":
    unittest.main()


class DatasetFlag(unittest.TestCase):
    """``--dataset <root>`` has to stand in for --images/--labels/--classes on any layout."""

    def _resolve(self, argv):
        from pglabel import cli, desktop
        args = cli.build_parser().parse_args(argv)
        return desktop.resolve_dataset_args(args), args

    def _make(self, tmp, shape):
        root = Path(tmp) / "ds"
        if shape == "split":
            _img(root / "images" / "train"), _lab(root / "labels" / "train")
            (root / "data.yaml").write_text("names: [cat, dog]\n")
        elif shape == "flat":
            _img(root / "images"), _lab(root / "labels")
            (root / "classes.txt").write_text("cat\ndog\n")
        elif shape == "unlabeled":
            _img(root / "images")
        return root

    def test_resolves_every_layout(self):
        for shape in ("split", "flat"):
            with tempfile.TemporaryDirectory() as tmp:
                root = self._make(tmp, shape)
                rc, a = self._resolve(["--dataset", str(root)])
                self.assertEqual(rc, 0, shape)
                self.assertTrue(Path(a.images).is_dir(), shape)
                self.assertIsNotNone(a.labels, shape)
                self.assertEqual(a.classes, "cat,dog", shape)

    def test_unlabeled_dataset_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, a = self._resolve(["--dataset", str(self._make(tmp, "unlabeled"))])
            self.assertEqual(rc, 0)
            self.assertIsNone(a.labels)
            self.assertEqual(a.classes, "object")

    def test_explicit_flags_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make(tmp, "split")
            rc, a = self._resolve(["--dataset", str(root), "--classes", "x,y"])
            self.assertEqual((rc, a.classes), (0, "x,y"))

    def test_folder_with_no_images_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, _ = self._resolve(["--dataset", tmp])
            self.assertEqual(rc, 2)

    def test_bare_images_flag_still_finds_the_class_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make(tmp, "flat")
            rc, a = self._resolve(["--images", str(root / "images")])
            self.assertEqual((rc, a.classes), (0, "cat,dog"))

    def test_none_classes_never_becomes_a_class_called_None(self):
        from pglabel import dataset_setup, state
        with tempfile.TemporaryDirectory() as tmp:
            dataset_setup.apply_dataset(Path(tmp), Path(tmp) / "lab", None)
            self.assertEqual(state.CFG["classes"], ["object"])
