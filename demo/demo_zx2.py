# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
from PIL import Image, ImageDraw, ImageFont

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"
COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument("--in_dir")
    # parser.add_argument("--out_dir")
    parser.add_argument("--out_info")
    parser.add_argument("--font_path", default='~/data_public/Songti.ttc')

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def detect_car(img):
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(img)
    logger.info(
        "{}: {} in {:.2f}s".format(
            path,
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )

    best_box, best_score, max_area = None, None, 0
    if "instances" in predictions:
        boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
        scores = predictions["instances"].scores if predictions["instances"].has("scores") else None
        classes = predictions["instances"].pred_classes if predictions["instances"].has("pred_classes") else None
        from detectron2.structures import Boxes
        if isinstance(boxes, Boxes):
            boxes = boxes.tensor.cpu().numpy()
        import torch
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(classes, torch.Tensor):
            classes = classes.cpu().numpy()
        for i, cls in enumerate(classes):
            if cls == 2:
                area = abs(boxes[i][2] - boxes[i][0]) * abs(boxes[i][3] - boxes[i][1])
                if best_box is None:
                    max_area, best_box, best_score = area, boxes[i], scores[i]
                elif area > max_area and scores[i] + 0.3 > best_score:
                    max_area, best_box, best_score = area, boxes[i], scores[i]

    return predictions, visualized_output, best_box, best_score, boxes, scores, classes


def get_map_of_all_img_path(in_dir):
    '''
    :param in_dir: 输入目录
    :return: 文件全路径->(相对子路径，文件名)
    '''
    img_path_map = {}
    for filename in os.listdir(in_dir):
        img_path = os.path.join(in_dir, filename)
        if os.path.isdir(img_path):
            s_img_path_map = get_map_of_all_img_path(img_path)
            for k, v in s_img_path_map.items():
                img_path_map[k] = [os.path.join(filename, v[0]), v[1]]
        elif os.path.isfile(img_path):
            name, suffix = os.path.splitext(os.path.basename(filename))
            if suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                print("{} : {} is unkndown suffix. ".format(filename, suffix))
                continue
            img_path_map[img_path] = ('', filename)
    return img_path_map


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    img_path_map = get_map_of_all_img_path(args.in_dir)
    logger.debug(" img_path_map size : {}".format(len(img_path_map)))

    fo = open(args.out_info, 'w')
    index = 0
    font = ImageFont.truetype(args.font_path, 20)
    for path, (sub_dir, filename) in img_path_map.items():
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")

        predictions, visualized_output, best_box, best_score, boxes, scores, classes = detect_car(img)
        # img = Image.open(path)

        out_dir = os.path.join(args.output, sub_dir)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        assert os.path.isdir(args.output), args.output
        # out_filename = os.path.join(args.output, os.path.basename(path))
        # visualized_output.save(out_filename)
        if best_score is not None and best_box is not None:
            x1, y1 = int(best_box[0]), int(best_box[1])
            x2, y2 = int(best_box[2] + 0.5), int(best_box[3] + 0.5)
            img_car_best = img[y1:y2, x1:x2, :]
            img.save(os.path.join(out_dir, filename))
            '''
            draw = ImageDraw.Draw(img)
            draw.rectangle(best_box.tolist(), outline=(0, 255, 0), width=6)
            fo.write("{}\t{}\t{}\t{}\n".format(os.path.join(sub_dir, filename),
                                                ",".join([str(x) for x in best_box.tolist()]),
                                                best_score,
                                                2))
            '''
        else:
            continue
            '''
            w, h = img.size
            fo.write("{}\t{}\t{}\t{}\n".format(os.path.join(sub_dir, filename),
                                                ",".join([str(x) for x in [0, 0, w, h]]),
                                                0,
                                                2))
            '''
        '''
        for i, box in enumerate(boxes):
            draw = ImageDraw.Draw(img)
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=2)
            draw.text((box[0], box[1]),
                        "{}.{}.{:.2f}".format(classes[i], COCO_CLASSES[classes[i]], scores[i]),
                        font=font,
                        fill=(0, 0, 255))
        img.save(os.path.join(out_dir, filename))
        '''
        # break
    fo.close()
