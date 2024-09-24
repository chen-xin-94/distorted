import numpy as np
import cv2
import os
from tqdm import tqdm
from pycocotools.coco import COCO
import socket
from pathlib import Path
import json
import copy
from utils import (
    load_camera_params_from_cfg_file,
    ltwh2xyxy,
    xyxy2ltwh,
    get_image_path,
    load_target,
    undistort_mesh,
    distort_image,
    get_bounding_rect,
    invert_map,
    extract_person_coco,
)

## specify constant
hostname = socket.gethostname()
if hostname == "hctlrds":
    COCO_FOLDER = "/mnt/ssd2/xin/data/coco/"
elif hostname == "Chen-Mac-mini.local":
    COCO_FOLDER = "/Volumes/Storage/Datasets/coco/"
else:
    raise NotImplementedError

splits = ["val2017", "train2017"]

camera_cfg = (
    "cfg/camera/2024_8_20_11_38_34_720x960_4149298_2024_08_02_A5PRT9_ip_72_test.cfg"
)

cam_matrix, dist_vec = load_camera_params_from_cfg_file(camera_cfg)
K = np.array(cam_matrix)
D = np.array(dist_vec)

coco_folder = COCO_FOLDER

for split in splits:
    root = os.path.join(coco_folder, split)
    annFile = os.path.join(coco_folder, "annotations", f"instances_{split}.json")

    coco = COCO(annFile)

    coco_distorted_folder = Path(coco_folder.replace("/coco/", "/coco_distorted/"))
    root_coco_distorted = Path(os.path.join(coco_distorted_folder, split))
    annFile_coco_distorted = Path(annFile.replace("/coco/", "/coco_distorted/"))
    if not root_coco_distorted.exists():
        root_coco_distorted.mkdir(parents=True, exist_ok=True)
        print(f"Created {root_coco_distorted}")
    if not annFile_coco_distorted.exists():
        annFile_coco_distorted.parent.mkdir(parents=True, exist_ok=True)
        print(f"Created {annFile_coco_distorted}")

    with open(annFile, "r") as f:
        data = json.load(f)

    images = []
    annotations = []
    excluded_images = set()
    for image_entry in tqdm(data["images"]):

        id = image_entry["id"]

        # load image and targets
        targets = load_target(coco, id)
        img_path = os.path.join(root, get_image_path(coco, id))

        # process image
        img = cv2.imread(img_path)

        u, v = undistort_mesh(img, K, D)
        distorted_img = distort_image(img, u, v)
        X, Y, W, H = get_bounding_rect(distorted_img)
        cropped_distorted_image = distorted_img[Y : Y + H, X : X + W]
        new_h, new_w = cropped_distorted_image.shape[:2]
        # update image entry for data_distorted
        image_entry["width"] = new_w
        image_entry["height"] = new_h
        images.append(image_entry)
        # save image
        new_img_path = os.path.join(root_coco_distorted, get_image_path(coco, id))
        cv2.imwrite(new_img_path, cropped_distorted_image)

        # process boxes
        uv = np.stack([u, v], axis=-1)
        uv_inverse = invert_map(uv)

        bboxes = []
        areas = []
        for target in targets:
            bbox = np.array(target["bbox"])
            bbox_xyxy = ltwh2xyxy(bbox)
            x1, y1, x2, y2 = bbox_xyxy.astype(int)
            # clamp the value of x1, y1, x2, y2
            x1, x2 = np.clip([x1, x2], 0, uv_inverse.shape[1] - 1)
            y1, y2 = np.clip([y1, y2], 0, uv_inverse.shape[0] - 1)
            distorted_bbox_xyxy = np.concatenate(
                (uv_inverse[y1, x1], uv_inverse[y2, x2])
            )

            cropped_distorted_bbox_xyxy = distorted_bbox_xyxy - np.array([X, Y, X, Y])

            # if any of the box coord is negative, skip this box
            # NOTE: this is caused by the original black border/scene of the image
            # which makes the get_bounding_rect() function capture the contour inside the distored image
            # instead of the distored image itself, see Visualization_problem.ipynb for a visualization
            if np.any(cropped_distorted_bbox_xyxy < 0):
                # mark the image as excluded
                excluded_images.add(id)
                continue

            cropped_distorted_bbox = xyxy2ltwh(cropped_distorted_bbox_xyxy)
            # update annotations
            annotations.append(
                {
                    "id": target["id"],
                    "image_id": id,
                    "category_id": target["category_id"],
                    "bbox": cropped_distorted_bbox.tolist(),
                    "area": cropped_distorted_bbox[2] * cropped_distorted_bbox[3],
                    "iscrowd": target["iscrowd"],
                    # TODO: drop segmentation for now
                    # "segmentation": target["segmentation"],
                }
            )

    # update annotations
    data_distorted = copy.deepcopy(data)
    data_distorted["images"] = images
    data_distorted["annotations"] = annotations
    with open(annFile_coco_distorted, "w") as f:
        json.dump(data_distorted, f)

    # save excluded images
    with open(
        str(annFile_coco_distorted).replace(".json", "_excluded_images.txt"), "w"
    ) as f:
        for image_id in excluded_images:
            f.write(f"{image_id}\n")

    # extract person
    extract_person_coco(annFile_coco_distorted)
