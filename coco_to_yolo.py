# modified from https://github.com/ultralytics/JSON2YOLO/blob/main/general_json2yolo.py

import os
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import json


def convert_coco_json(
    json_dir="/mnt/ssd2/xin/data/coco_distorted/annotations",
    original_coco_dir="/mnt/ssd2/xin/data/coco/",
    new_dir="new_dir/",
    use_segments=False,
    cls91to80=True,
):
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping."""
    save_dir = make_dirs(new_dir, original_coco_dir)  # output directory
    coco80 = coco91_to_coco80_class()

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        fn = (
            Path(save_dir) / "labels" / json_file.stem.replace("instances_", "")
        )  # folder name
        fn.mkdir()
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {"%g" % x["id"]: x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images["%g" % img_id]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = (
                    coco80[ann["category_id"] - 1]
                    if cls91to80
                    else ann["category_id"] - 1
                )  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                # Segments
                if use_segments:
                    if len(ann["segmentation"]) > 1:
                        s = merge_multi_segment(ann["segmentation"])
                        s = (
                            (np.concatenate(s, axis=0) / np.array([w, h]))
                            .reshape(-1)
                            .tolist()
                        )
                    else:
                        s = [
                            j for i in ann["segmentation"] for j in i
                        ]  # all segments concatenated
                        s = (
                            (np.array(s).reshape(-1, 2) / np.array([w, h]))
                            .reshape(-1)
                            .tolist()
                        )
                    s = [cls] + s
                    if s not in segments:
                        segments.append(s)

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (
                        *(segments[i] if use_segments else bboxes[i]),
                    )  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")


def generate_image_list_txt(
    annotation_dir,
    new_dir="/mnt/ssd2/xin/data/coco_distorted/annotations",
):
    """
    Generate image list from json files.
    Args:
        annotation_dir: str, directory of the to be convertedjson files.
        new_dir: str, directory to save the image list.
    """

    # exclude following images, mainly because their original black edges break the get_bounding_rect()
    # so that the final distorted images are not cropped correctly
    exclude_image_ids = [
        5356,
        14337,
        38986,
        41666,
        47509,
        53404,
        58511,
        62790,
        64933,
        68260,
        72002,
        82157,
        83519,
        95875,
        103033,
        114629,
        121519,
        129436,
        137134,
        139604,
        143629,
        185335,
        186738,
        234515,
        245337,
        286353,
        287560,
        303408,
        314791,
        314996,
        337915,
        344507,
        349090,
        351217,
        363546,
        365575,
        370793,
        388531,
        429614,
        430341,
        430791,
        432702,
        434786,
        469870,
        475960,
        487943,
        491061,
        519211,
        530383,
        551042,
        557291,
        566757,
        569723,
        569867,
        570416,
        580971,
        581732,
        232684,
        233370,
        433915,
        477805,
    ]

    for json_file in Path(annotation_dir).glob("*.json"):
        if "_val" in json_file.stem:
            split = "val2017"
        elif "_train" in json_file.stem:
            split = "train2017"
        else:
            raise ValueError(f"Unknown split in {json_file}")
        with open(json_file) as f:
            data = json.load(f)
        filenames = [
            f"./images/{split}/{x['file_name']}"
            for x in data["images"]
            if x["id"] not in exclude_image_ids
        ]
        # delete instances_ prefix
        txt_path = Path(new_dir) / (json_file.stem.replace("instances_", "") + ".txt")
        with open(txt_path, "w") as f:
            for filename in filenames:
                f.write(filename + "\n")
        print(f"Saved {txt_path}")


def make_dirs(dir="new_dir/", original_coco_dir="/mnt/ssd2/xin/data/coco/"):
    """Creates a directory with subdirectories 'labels' and 'images' and links to COCO images."""
    dir = Path(dir)
    if dir.exists():
        raise FileExistsError(f"Output directory '{dir}' already exists")
    for p in dir, dir / "labels", dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir
    for split in ["train2017", "val2017", "test2017"]:
        os.symlink(
            original_coco_dir + split,
            dir / "images" / split,
            target_is_directory=True,
        )
    return dir


def coco91_to_coco80_class():  # converts 80-index (val2014) to 91-index (paper)
    """Converts COCO 91-class index (paper) to 80-class index (2014 challenge)."""
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]


def merge_multi_segment(segments):
    """
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).

    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def generate_person_image_folder(coco_dir_yolo, coco_dir):
    """
    Generate image folder corresponds to labels/train2017_person and labels/val2017_person given already generated image list txt.
    This function force the txt image list and the image folder share the same name, otherwise the yolo training will fail.
    """
    coco_dir_yolo = Path(coco_dir_yolo)
    for split in ["val2017_person", "train2017_person"]:
        image_folder = coco_dir_yolo / "images" / split
        image_folder.mkdir(parents=True, exist_ok=True)
        with open(coco_dir_yolo / (split + ".txt")) as f:
            image_list = f.readlines()
        new_image_list = []
        for image in tqdm(image_list):
            old_image_path = Path(coco_dir) / "/".join(image.strip().split("/")[2:])
            new_image_path = image_folder / Path(old_image_path).name
            # print(old_image_path, new_image_path)
            new_image_list.append(new_image_path)
            os.symlink(old_image_path, new_image_path)
            # shutil.copy(old_image_path, new_image_path)
        with open(coco_dir_yolo / (split + ".txt"), "w") as f:
            for image in new_image_list:
                f.write("./" + str(image.relative_to(coco_dir_yolo)) + "\n")


if __name__ == "__main__":
    # distorted coco dataset in yolo format
    new_dir = "/mnt/ssd2/xin/datasets/coco_distorted/"
    # distorted coco dataset dir
    coco_dir = "/mnt/ssd2/xin/data/coco_distorted/"
    # distorted coco dataset annotation dir
    json_dir = "/mnt/ssd2/xin/data/coco_distorted/annotations"
    # raw coco dataset
    original_coco_dir = "/mnt/ssd2/xin/data/coco/"

    # convert_coco_json(json_dir, original_coco_dir, new_dir)
    # generate_image_list_txt(json_dir, new_dir)
    generate_person_image_folder(new_dir, coco_dir)
