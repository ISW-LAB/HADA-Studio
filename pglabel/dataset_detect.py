#!/usr/bin/env python3
"""Recognising somebody else's dataset folder.

The start screen used to accept exactly one shape — ``<name>/images/train`` plus
``<name>/labels/train`` plus a ``*.yaml`` naming the classes — and silently skipped anything else.
That is stricter than reality: the bundled sample under ``data/`` is ``images/`` + ``labels/`` +
``classes.txt``, and would not have been offered as a preset by the very app that ships it.

So this module answers two questions about an arbitrary folder a user points at:

    where are the images and labels?   ``resolve_layout`` — tries the layouts people actually
                                       have, most specific first, and accepts an images-only
                                       folder, which for an AUTO-labeling tool is the normal
                                       starting point rather than an error.
    what are the classes called?       ``resolve_class_names`` — a data.yaml in any of its three
                                       spellings, else classes.txt / obj.names, else names read
                                       off the label files themselves so the ids at least line up.

Nothing here imports anything outside the standard library: it runs inside the app process, which
deliberately carries no third-party dependency beyond Pillow.
"""

from __future__ import annotations

import re
from pathlib import Path

from .labelio import is_image

# Where images and labels sit, relative to the dataset root. Ordered most-specific first so a
# dataset that has BOTH images/train and images/ resolves to the split it clearly intends.
LAYOUTS = [
    ("images/train", "labels/train"),      # ultralytics split layout
    ("train/images", "train/labels"),      # roboflow export
    ("images", "labels"),                  # the flat layout, and our own data/ sample
    ("JPEGImages", "labels"),              # VOC-style images beside YOLO txt
    (".", "labels"),                       # images loose in the root
]
CLASS_FILES = ("classes.txt", "obj.names", "class.names", "names.txt")


def _count_images(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and is_image(p))


def resolve_layout(root) -> dict | None:
    """Find the images (and, if present, labels) folder of a dataset root.

    Returns ``{"images", "labels", "layout", "n_images", "n_labels"}``, or None when the folder
    holds no images at all under any known layout. ``labels`` is None for an unlabeled dataset —
    that is a legitimate dataset for this tool, not a failure.
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        return None
    best = None
    for img_rel, lab_rel in LAYOUTS:
        img = (root / img_rel).resolve() if img_rel != "." else root.resolve()
        n = _count_images(img)
        if n == 0:
            continue
        lab = root / lab_rel
        n_lab = len(list(lab.glob("*.txt"))) if lab.is_dir() else 0
        cand = {"images": img, "labels": lab.resolve() if lab.is_dir() else None,
                "layout": f"{img_rel} + {lab_rel}" if lab.is_dir() else f"{img_rel} (unlabeled)",
                "n_images": n, "n_labels": n_lab}
        # A layout that also found labels beats one that only found images, whatever the order.
        if best is None or (cand["n_labels"] > 0 and best["n_labels"] == 0):
            best = cand
        if best["n_labels"] > 0:
            break
    return best


def parse_yaml_class_names(yaml_path) -> list:
    """Pull class names out of an ultralytics data.yaml.

    Three spellings occur in the wild and all three are handled: the inline list
    (``names: [car, van]``), the index map (``names:\\n  0: car``) and the dash list
    (``names:\\n  - car``), which YOLOv5-era exports still use. Parsed with regular expressions
    rather than a YAML dependency, for the no-third-party reason in the module docstring.
    """
    try:
        text = Path(yaml_path).read_text(encoding="utf-8-sig")
    except Exception:
        return []
    m = re.search(r'^\s*names:\s*\[(.*?)\]', text, re.M | re.S)       # inline list
    if m:
        return [x.strip().strip('\'"') for x in m.group(1).split(',') if x.strip()]
    indexed, dashed, in_names = {}, [], False                         # block forms
    for line in text.splitlines():
        if re.match(r'^\s*names:\s*$', line):
            in_names = True
            continue
        if not in_names:
            continue
        mm = re.match(r'^\s+(\d+)\s*:\s*(.+?)\s*$', line)             # "  0: car"
        if mm:
            indexed[int(mm.group(1))] = mm.group(2).strip().strip('\'"')
            continue
        md = re.match(r'^\s+-\s*(.+?)\s*$', line)                     # "  - car"
        if md:
            dashed.append(md.group(1).strip().strip('\'"'))
            continue
        if line.strip() and not line[0].isspace():                    # next top-level key
            break
    if indexed:
        return [indexed[i] for i in sorted(indexed)]
    return dashed


def _names_from_file(path: Path) -> list:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except Exception:
        return []
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def _max_class_id(labels_dir: Path, cap: int = 400) -> int:
    """Highest class id appearing in the label files — the fallback when nothing names them."""
    top = -1
    for i, f in enumerate(sorted(labels_dir.glob("*.txt"))):
        if i >= cap:                                     # a sample is enough to size the list
            break
        try:
            for ln in f.read_text(encoding="utf-8-sig").splitlines():
                parts = ln.split()
                if parts:
                    top = max(top, int(float(parts[0])))
        except Exception:
            continue
    return top


def resolve_class_names(root, labels_dir=None) -> tuple:
    """Class names for a dataset, and where they came from.

    Returns ``(names, source)``. ``source`` is worth surfacing in the UI: a user who sees
    "class0, class1" and the word *labels* understands immediately that they should supply a
    data.yaml, whereas silently inventing names looks like the tool guessed wrong.
    """
    root = Path(root).expanduser()
    for y in sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml")):
        names = parse_yaml_class_names(y)
        if names:
            return names, y.name
    for cf in CLASS_FILES:
        names = _names_from_file(root / cf)
        if names:
            return names, cf
    if labels_dir and Path(labels_dir).is_dir():
        top = _max_class_id(Path(labels_dir))
        if top >= 0:
            return [f"class{i}" for i in range(top + 1)], "labels"
    return [], ""


def describe(root) -> dict | None:
    """Everything the start screen needs about one candidate dataset folder, or None."""
    lay = resolve_layout(root)
    if lay is None:
        return None
    names, src = resolve_class_names(root, lay["labels"])
    return {**lay, "name": Path(root).expanduser().resolve().name,
            "root": str(Path(root).expanduser().resolve()),
            "classes": names, "class_source": src}
