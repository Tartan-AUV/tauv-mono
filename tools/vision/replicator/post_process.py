import torch
import torchvision.transforms.v2 as T
import argparse
import pathlib
import glob
import random
from PIL import Image
import re
import json
import numpy as np
from typing import Dict
from functools import partial
from multiprocessing import Pool
import matplotlib.pyplot as plt

from tauv_vision.datasets.segmentation_dataset.segmentation_dataset import SegmentationSample


def get_id(path: pathlib.Path) -> str:
    match = re.search(r"_(\d+)\.", str(path))
    if match:
        id = match.group(1)
        return id
    else:
        raise ValueError("no match for id")


def parse_seg_value(s: str) -> (int, int, int, int):
    return [int(x) for x in s[1:-1].split(",")]


def post_process(rgb_path: pathlib.Path, background_path: pathlib.Path,
                 in_dir: pathlib.Path, background_dir: pathlib.Path, out_dir: pathlib.Path, class_names: Dict[str, int]):
    id = get_id(rgb_path)

    seg_path = (in_dir / f"instance_segmentation_{id}").with_suffix(".png")
    seg_instance_path = (in_dir / f"instance_segmentation_mapping_{id}").with_suffix(".json")

    bbox_path = (in_dir / f"bounding_box_2d_loose_{id}").with_suffix(".npy")
    bbox_classification_path = (in_dir / f"bounding_box_2d_loose_labels_{id}").with_suffix(".json")
    bbox_instance_path = (in_dir / f"bounding_box_2d_loose_prim_paths_{id}").with_suffix(".json")

    depth_path = (in_dir / f"distance_to_camera_{id}").with_suffix(".npy")

    img_pil = Image.open(rgb_path)
    background_pil = Image.open(background_path)
    seg_pil = Image.open(seg_path)
    seg_raw = T.ToTensor()(seg_pil)
    depth_np = np.load(depth_path)

    background_np = np.array(background_pil).astype(np.float32) / 255
    img_np = np.array(img_pil)
    img_rgb_np = img_np[:, :, 0:3].astype(np.float32) / 255
    img_a_np = img_np[:, :, 3].astype(np.float32) / 255

    background_lighting_np = np.array([np.mean(background_np[:, :, 0]), np.mean(background_np[:, :, 1]), np.mean(background_np[:, :, 2])]) + np.random.uniform(-0.05, 0.05, (3,))
    beta = np.random.uniform(0.1, 0.2)

    transmission_np = np.maximum(np.exp(-beta * depth_np), 0.1)
    img_rgb_adj_np = np.expand_dims(transmission_np, 2) * img_rgb_np + np.expand_dims(1 - transmission_np, 2) * background_lighting_np

    composite_np = np.expand_dims(img_a_np, 2) * img_rgb_adj_np + np.expand_dims(1 - img_a_np, 2) * background_np

    composite_pil = Image.fromarray((composite_np * 255).astype(np.uint8), "RGB")

    img = T.ToTensor()(composite_pil)

    w, h = img_pil.size

    bboxes = np.load(bbox_path)
    with open(bbox_classification_path, "r") as fp:
        bbox_classifications = json.load(fp)
    with open(bbox_instance_path, "r") as fp:
        bbox_instances = json.load(fp)
    with open(seg_instance_path, "r") as fp:
        seg_instances = json.load(fp)

    n_detections = len(bboxes)

    seg_instances = {v: k for k, v in seg_instances.items()}

    # valid = torch.full((n_detections,), fill_value=True, dtype=torch.bool)
    # classifications = torch.zeros(n_detections, dtype=torch.long)
    # bounding_boxes = torch.zeros((n_detections, 4), dtype=torch.float)
    valid = []
    classifications = []
    bounding_boxes = []
    seg = torch.full((h, w), fill_value=255, dtype=torch.uint8)

    detection_i = 0
    for i in range(len(bboxes)):
        bbox_class, x0, y0, x1, y1, _ = bboxes[i]

        bbox_x = ((x0 + x1) / 2) / w
        bbox_y = ((y0 + y1) / 2) / h

        bbox_w = abs(x1 - x0) / w
        bbox_h = abs(y1 - y0) / h

        bbox_class_name = bbox_classifications[str(bbox_class)]["class"].split(",")[-1]

        if bbox_class_name not in class_names:
            continue

        class_id = class_names[bbox_class_name]

        valid.append(True)
        classifications.append(class_id)
        bounding_boxes.append([bbox_y, bbox_x, bbox_h, bbox_w])

        if bbox_instances[i] in seg_instances:
            seg_value = parse_seg_value(seg_instances[bbox_instances[i]])
            seg_mask = seg_raw == (torch.Tensor(seg_value).unsqueeze(1).unsqueeze(2) / 255)

            seg[seg_mask[0] & seg_mask[1] & seg_mask[2] & seg_mask[3]] = detection_i

        detection_i += 1

    valid = torch.Tensor(valid).to(torch.bool)
    classifications = torch.Tensor(classifications).to(torch.long)
    bounding_boxes = torch.Tensor(bounding_boxes).to(torch.float)

    sample = SegmentationSample(
        img=img,
        seg=seg,
        valid=valid,
        classifications=classifications,
        bounding_boxes=bounding_boxes,
    )

    out_id = id.zfill(8)
    sample.save(out_dir, out_id)

def f(rgb_path, background_paths, in_dir, background_dir, out_dir, class_names):
    background_path = random.choice(background_paths)
    post_process(rgb_path, background_path, in_dir, background_dir, out_dir, class_names)


def run(in_dir: pathlib.Path, background_dir: pathlib.Path, out_dir: pathlib.Path):
    rgb_paths = glob.glob("rgb_*.png", root_dir=in_dir)
    rgb_paths = [in_dir / rgb_path for rgb_path in rgb_paths]

    background_paths = glob.glob("*.png", root_dir=background_dir)
    background_paths = [background_dir / background_path for background_path in background_paths]

    # TODO: READ CLASS NAMES
    class_names = {"torpedo_22_circle": 0, "torpedo_22_trapezoid": 1}

    # for rgb_path in rgb_paths:
    #     f(rgb_path, background_paths, in_dir, background_dir, out_dir, class_names)

    with Pool() as pool:
        f_partial = partial(f, background_paths=background_paths, in_dir=in_dir, background_dir=background_dir, out_dir=out_dir, class_names=class_names)
        pool.map(f_partial, rgb_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir")
    parser.add_argument("background_dir")
    parser.add_argument("out_dir")

    args = parser.parse_args()

    in_dir = pathlib.Path(args.in_dir).expanduser()
    background_dir = pathlib.Path(args.background_dir).expanduser()
    out_dir = pathlib.Path(args.out_dir).expanduser()

    assert in_dir.is_dir()
    assert background_dir.is_dir()

    if not out_dir.exists():
        out_dir.mkdir()

    run(in_dir, background_dir, out_dir)


if __name__ == "__main__":
    main()