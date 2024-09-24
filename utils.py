import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict, List, Any
from pathlib import Path
from PIL import Image
import cv2
from ultralytics.utils.plotting import Annotator, Colors
import os
from pycocotools.coco import COCO
import json

COLORS = Colors()


def load_camera_params_from_cfg_file(filename, r_t_vecs=False):
    """function that reads out intrinsic camera parameters from cfg file.
    file is formatted according to needs of TES Bird's Eye.
    return params:
    cam_values_matrix: matrix with intrinsic values fx, fy, cx, cy
    dist_values_vector: vector with distortion parameters
    if r_t_vecs is True:
    tvecs: extrinsic translational parameters
    r_vec: rotational parameters
    """

    with open(filename, "r") as f:
        lines = f.read().splitlines()

    # Extracting cam_values_string and converting it back to matrix
    cam_values_line = [line for line in lines if line.startswith("MAT:CAM:VALUES=")][0]
    cam_values_string = cam_values_line.split("=")[1]
    cam_values_list = [float(value) for value in cam_values_string.split("#")]
    cam_values_matrix = [
        cam_values_list[i : i + 3] for i in range(0, len(cam_values_list), 3)
    ]

    # Extracting dist_values_string and converting it back to vector
    dist_values_line = [line for line in lines if line.startswith("MAT:DIST:VALUES=")][
        0
    ]
    dist_values_string = dist_values_line.split("=")[1]
    # each element of distortion vector represents list to be in accordance
    # with output of calibrate_intrinsics function of opencv
    dist_values_vector = [float(value) for value in dist_values_string.split("#")]

    if r_t_vecs is True:
        tvecs = []
        # Extracting tvec_values_string and converting it back to vector
        tvec_line = [line for line in lines if line.startswith("tvec_")][0]
        tvec_string = tvec_line.split("=")[1]
        # each element of tvec vector represents list to be in accordance
        # with output of calibrate_intrinsics function of opencv
        tvecs.append([float(value) for value in tvec_string.split("#")])

        rvecs = []
        # Extracting rvec_values_string and converting it back to vector
        rvec_line = [line for line in lines if line.startswith("rvec_")][0]
        rvec_string = rvec_line.split("=")[1]
        # each element of rvec vector represents list to be in accordance
        # with output of calibrate_intrinsics function of opencv
        rvecs.append([float(value) for value in rvec_string.split("#")])

        return cam_values_matrix, dist_values_vector, tvecs, rvecs
    else:
        return cam_values_matrix, dist_values_vector


def ltwh2xyxy(x):
    """
    It converts the bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.

    Args:
        x (np.ndarray | torch.Tensor): the input image

    Returns:
        y (np.ndarray | torch.Tensor): the xyxy coordinates of the bounding boxes.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]  # width
    y[..., 3] = x[..., 3] + x[..., 1]  # height
    return y


def xyxy2ltwh(x):
    """
    It converts the bounding box from [x1, y1, x2, y2] to [x1, y1, w, h] where xy1=top-left, xy2=bottom-right.

    Args:
        x (np.ndarray | torch.Tensor): the input image

    Returns:
        y (np.ndarray | torch.Tensor): the ltwh coordinates of the bounding boxes.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def annotate_image(
    img: Union[torch.Tensor, np.ndarray],
    boxes: Union[torch.Tensor, np.ndarray],
    category: Union[torch.Tensor, np.ndarray],
    confs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    category_id_to_name: Dict[int, str] = None,
    colors=COLORS,
    conf_thres: float = 0.5,
    masks: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Optional[Union[Path, str]] = None,
    line_width: Optional[float] = None,
    font_size: Optional[float] = None,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Plot image grid with labels, bounding boxes, and masks, using ultralytics Annotator class.

    Args:
        img: one singel image to plot on. Shape: (channels, height, width).
        boxes: Bounding boxes for each detection. Shape: (num_detections, 4). Format: (x1, y1, x2, y2).
        category: Class id for each detection. Shape: (num_detections,).
        category_id_to_name: Dictionary mapping class id to class name.
        colors: Colors object for mapping class indices to colors.
        confs: Confidence scores for each detection. Shape: (num_detections,).
        conf_thres: Confidence threshold for displaying detections.
        masks: Instance segmentation masks. Shape: (num_detections, height, width)
        paths: List of file paths for each image in the batch.
        save_path: File path to save the plotted image.
        line_width: Line width for bounding boxes.
        font_size: Font size for class labels.
        alpha: Alpha value for masks.

    Returns:
        np.ndarray: Plotted image as a numpy array

    Note:
        This function supports both tensor and numpy array inputs. It will automatically
        convert tensor inputs to numpy arrays for processing.
    """

    if isinstance(img, torch.Tensor):
        img = img.cpu().float().numpy()
    if isinstance(category, torch.Tensor):
        category = category.cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)

    # whether to show ground truth labels (without confidence) or predictions (with confidence)
    is_gt = confs is None

    # init annotator
    annotator = Annotator(im=img.copy(), line_width=line_width, font_size=font_size)

    for j, box in enumerate(boxes.astype(np.int64).tolist()):
        c = category[j]
        color = colors(c)
        c = category_id_to_name.get(c, c) if category_id_to_name else c

        if is_gt or confs[j] > conf_thres:

            # draw box with label and conf
            label = f"{c}" if is_gt else f"{c} {confs[j]:.2f}"
            annotator.box_label(box, label, color=color)

            # draw mask
            if masks is not None:
                mask = masks[j]
                annotator.im[mask == 1] = (
                    annotator.im[mask == 1] * (1 - alpha) + np.array(color) * alpha
                )
    if save_path:
        annotator.im.save(save_path)
    return annotator.result()


def display_image(input, figsize=(12, 12)):
    """Display a numpy array or a torch.tensor as an image"""
    # tensor to array
    if isinstance(input, torch.Tensor):
        input = input.cpu().detach().numpy()
    # CHW to HWC
    shape = input.shape
    if len(shape) == 4 and shape[1] in [1, 3]:
        input = np.transpose(input, (0, 2, 3, 1))[0]
    if len(shape) == 3 and shape[0] in [1, 3]:
        input = np.transpose(input, (1, 2, 0))

    plt.figure(figsize=figsize)  # Set the figure size for better visibility
    plt.imshow(
        input, cmap="gray"
    )  # Display the image, you can specify cmap for grayscale if needed
    plt.axis("off")  # Turn off axis labels
    plt.show()  # Render the image


def get_image_path(coco, id: int) -> str:
    return coco.loadImgs(id)[0]["file_name"]


def load_target(coco, id: int) -> List[Any]:
    return coco.loadAnns(coco.getAnnIds(id))


def undistort_points(x, y, K, D):
    """
    undistort points using the fisheye model.
    follow https://docs.opencv2.org/3.4/db/d58/group__calib3d__fisheye.html,
    but use an inverse scale

    Args:
    - x: x coordinate of the point, 1d or 2d numpy array
    - y: y coordinate of the point, 1d or 2d numpy array
    - K: camera matrix
    - D: distortion coefficients
    """

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, k3, k4 = D
    a = (x - cx) / fx
    b = (y - cy) / fy
    r = np.sqrt(a * a + b * b)
    theta = np.arctan(r)
    theta_d = theta * (
        1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8
    )

    # opencv uses the inverse of this scale
    scale = r / theta_d
    # scale = theta / r

    x_distorted = a * scale
    y_distorted = b * scale

    u = x_distorted * fx + cx
    v = y_distorted * fy + cy

    return u.astype(np.float32), v.astype(np.float32)


def undistort_mesh(img, K, D):
    """
    undistort the mesh grid of the image
    Args:
    - img: from cv2.imread()
    - K: camera matrix
    - D: distortion coefficients
    """

    h, w = img.shape[:2]

    # resize camera matrix
    new_K = K.copy()
    new_K[0, :] = new_K[0, :] * w / 960
    new_K[1, :] = new_K[1, :] * h / 720

    # get mesh grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # undistort mesh grid
    u, v = undistort_points(u, v, new_K, D)

    return u.astype(np.float32), v.astype(np.float32)


def distort_image(img, u, v):
    """
    distort image using the undistorted mesh grid
    remap() does the lookup from output to intput. For every pixel in the target image, it looks up where it comes from in the source image, and then assigns an interpolated value.

    """
    return cv2.remap(img, u, v, cv2.INTER_LINEAR)


def get_bounding_rect(img):
    """
    Find the bounding rectangle of the non-black regions of the image,
    return the x, y, w, h of the bounding rectangle
    """

    # Convert to grayscale to find contours
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the contours of the non-black regions
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours by area
    max_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # Find the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(max_contour)

    return x, y, w, h


def invert_map(F):
    """
    inverse of cv2.remap()
    https://github.com/opencv/opencv/issues/22120
    """

    # shape is (h, w, 2), an "xymap"
    (h, w) = F.shape[:2]
    I = np.zeros_like(F)
    I[:, :, 1], I[:, :, 0] = np.indices((h, w))  # identity map
    P = np.copy(I)
    for i in range(10):
        correction = I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
        P += correction * 0.5
    return P


def extract_person_coco(
    annotation_path, save_folder="", save_name="", truncate_length=None
):
    coco = COCO(annotation_path)
    category_ids = coco.getCatIds(catNms=["person"])

    image_ids = coco.getImgIds(catIds=category_ids)
    if truncate_length:
        image_ids = image_ids[:truncate_length]
    annotation_ids = coco.getAnnIds(
        imgIds=image_ids, catIds=category_ids, iscrowd=False
    )

    print("number of images: {}".format(len(image_ids)))
    print("number of annotations: {}".format(len(annotation_ids)))

    images_person = coco.loadImgs(image_ids)
    annotations_person = coco.loadAnns(annotation_ids)

    new_dataset = coco.dataset
    new_dataset["images"] = images_person
    new_dataset["annotations"] = annotations_person

    if not save_folder:
        save_folder = os.path.dirname(annotation_path)

    if not save_name:
        save_name = os.path.basename(annotation_path).split(".")[0] + "_person.json"
        if truncate_length:
            save_name = save_name.replace(".json", "_{}.json".format(truncate_length))

    save_path = os.path.join(save_folder, save_name)
    with open(save_path, "w") as f:
        json.dump(new_dataset, f)
        print("saved to {}".format(save_path))
