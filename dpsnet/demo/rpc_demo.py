# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import glob
import json
import os
import random
import shutil
from collections import defaultdict

import cv2
from PIL import ImageFont
from tqdm import tqdm
from vizer.draw import draw_boxes

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


def main():
    parser = argparse.ArgumentParser(description="DPNet Demo")
    parser.add_argument(
        "--config-file",
        default="configs/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_density_map.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        type=str,
        help="path to images file",
    )
    parser.add_argument(
        "--save_dir",
        default='rpc_results',
        type=str,
        help="path to images file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
             "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.mkdir(args.save_dir)

    with open('/data7/lufficc/rpc/instances_test2019.json') as fid:
        data = json.load(fid)

    images = {}
    for x in data['images']:
        images[x['id']] = x

    annotations = defaultdict(list)
    for x in data['annotations']:
        annotations[images[x['image_id']]['file_name']].append(x)
    annotations = dict(annotations)

    counter = {
        'easy': 0,
        'medium': 0,
        'hard': 0,
    }

    data_images = data['images'].copy()
    random.shuffle(data_images)
    FONT = ImageFont.truetype('/data7/lufficc/projects/DPNet/demo/arial.ttf', 8)
    for image_ann in data_images:
        if counter[image_ann['level']] >= 20:
            continue
        image_path = os.path.join(args.images_dir, image_ann['file_name'])
        img = cv2.imread(image_path)
        annotation = annotations[image_ann['file_name']]
        prediction = coco_demo.run_on_opencv_image(img)

        new_size = (400, 400)

        img = cv2.resize(img, new_size)
        prediction = prediction.resize(new_size)

        boxes = prediction.bbox.numpy()
        labels = prediction.get_field('labels').numpy()
        scores = prediction.get_field('scores').numpy()

        img = draw_boxes(img, boxes, labels, scores, COCODemo.CATEGORIES, width=2, font=FONT, alpha=0.4)
        gt_labels = sorted([ann['category_id'] for ann in annotation])
        if gt_labels == sorted(labels.tolist()):
            print('Get {}.'.format(image_ann['level']))
            cv2.imwrite(os.path.join(args.save_dir, image_ann['level'] + '_' + os.path.basename(image_path)), img)
            counter[image_ann['level']] += 1


if __name__ == "__main__":
    main()
