import json
from pathlib import Path


def convert_split(split_dir: Path, class_map: dict):
    """
    Convert COCO segmentation JSON in `split_dir/_annotations.coco.json` to YOLO segmentation labels.
    - Writes labels to `split_dir/labels/<image_basename>.txt`
    - Each line: `cls x1 y1 x2 y2 ...` with coords normalized to [0,1]
    - Handles polygon-type segmentations; skips RLE segmentations.
    """
    anno_path = split_dir / "_annotations.coco.json"
    if not anno_path.exists():
        print(f"❌ Missing COCO JSON: {anno_path}")
        return

    with anno_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data.get("images", [])}
    # Build image file_name → id map as well
    image_by_name = {img["file_name"]: img for img in data.get("images", [])}

    labels_dir = split_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    # Init empty labels for all images to ensure files exist
    for img in images.values():
        (labels_dir / (Path(img["file_name"]).stem + ".txt")).write_text("", encoding="utf-8")

    annos = data.get("annotations", [])

    for ann in annos:
        image_id = ann.get("image_id")
        img = images.get(image_id)
        if not img:
            # try by file name if provided
            file_name = ann.get("file_name")
            img = image_by_name.get(file_name) if file_name else None
            if not img:
                continue

        w, h = img["width"], img["height"]
        cat_id = ann.get("category_id")
        if cat_id not in class_map:
            # unseen class, skip
            continue
        cls = class_map[cat_id]

        seg = ann.get("segmentation")
        if not seg:
            continue

        # COCO polygon: list of lists, flatten each polygon into normalized pairs
        # Skip RLE style (dict)
        if isinstance(seg, dict):
            # RLE segmentation unsupported here
            continue

        # Some datasets provide a single list; normalize structure to list-of-polygons
        polygons = seg if isinstance(seg, list) else [seg]

        # Write one line per polygon
        label_path = labels_dir / (Path(img["file_name"]).stem + ".txt")
        with label_path.open("a", encoding="utf-8") as lf:
            for poly in polygons:
                if not isinstance(poly, list) or len(poly) < 6:
                    # need at least 3 points
                    continue
                # poly is [x1,y1,x2,y2,...] in pixel coords
                coords = []
                # iterate pairs
                for i in range(0, len(poly), 2):
                    x = poly[i] / w
                    y = poly[i + 1] / h
                    # clamp to [0,1]
                    x = min(max(x, 0.0), 1.0)
                    y = min(max(y, 0.0), 1.0)
                    coords.extend([x, y])
                # Write: cls followed by coords
                line = "{} ".format(cls) + " ".join(f"{v:.6f}" for v in coords)
                lf.write(line + "\n")


def build_class_map(categories):
    """
    Build mapping from COCO category_id → YOLO class index [0..nc-1]
    """
    # Sort by id ascending for stable mapping
    sorted_cats = sorted(categories, key=lambda c: c.get("id"))
    id_to_idx = {c["id"]: i for i, c in enumerate(sorted_cats)}
    names = [c.get("name", f"cls{i}") for i, c in enumerate(sorted_cats)]
    return id_to_idx, names


def convert_root(root: Path):
    """
    Convert both train and valid splits under `root` which contains
    `_annotations.coco.json` and images.
    """
    train_json = root / "train" / "_annotations.coco.json"
    valid_json = root / "valid" / "_annotations.coco.json"
    test_json = root / "test" / "_annotations.coco.json"

    if not train_json.exists():
        print(f"❌ Missing {train_json}")
        return

    with train_json.open("r", encoding="utf-8") as f:
        train_data = json.load(f)

    # Build class map from train categories
    class_map, names = build_class_map(train_data.get("categories", []))
    print(f"✓ Classes: {names}")

    # Convert train
    convert_split(root / "train", class_map)
    # Convert valid if exists
    if valid_json.exists():
        convert_split(root / "valid", class_map)
    else:
        print("⚠️ No valid/_annotations.coco.json found; converted train only.")

    # Convert test if exists
    if test_json.exists():
        convert_split(root / "test", class_map)
    else:
        print("⚠️ No test/_annotations.coco.json found; skipped test.")

    # Update data.yaml with names/nc if present
    data_yaml = root / "data.yaml"
    if data_yaml.exists():
        try:
            content = data_yaml.read_text(encoding="utf-8")
            # naive replace nc/names blocks
            lines = []
            for line in content.splitlines():
                if line.strip().startswith("nc:"):
                    lines.append(f"nc: {len(names)}")
                elif line.strip().startswith("names:"):
                    names_str = ", ".join(f"'{n}'" for n in names)
                    lines.append(f"names: [{names_str}]")
                elif line.strip().startswith("test:"):
                    # leave existing test line as-is
                    lines.append(line)
                else:
                    lines.append(line)
            data_yaml.write_text("\n".join(lines), encoding="utf-8")
            print(f"✓ Updated {data_yaml} with nc/names")
        except Exception as e:
            print(f"⚠️ Failed to update {data_yaml}: {e}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "dataset" / "instance_segmentation"
    convert_root(root)
