# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import json
import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import Heatmap
from maskrcnn_benchmark.utils.density import contiguous_coco_category_to_super_category

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


def generate_density_map(labels, boxes, scale, size, num_classes=1, min_sigma=1.):
    height, width = size
    scale_h, scale_w = scale
    density_map = np.zeros((num_classes, height, width), dtype=np.float32)
    for category, box in zip(labels, boxes):
        x1, y1, x2, y2 = box
        x1 *= scale_w
        x2 *= scale_w
        y1 *= scale_h
        y2 *= scale_h
        w, h = x2 - x1, y2 - y1
        box_radius = min(w, h) / 2
        sigma = max(min_sigma, box_radius * 5 / (4 * 3))  # 3/5 of gaussian kernel is in box
        cx, cy = round((x1 + x2) / 2), round((y1 + y2) / 2)
        density = np.zeros((height, width), dtype=np.float32)
        density[min(cy, height - 1), min(cx, width - 1)] = 1
        density = ndimage.filters.gaussian_filter(density, sigma, mode='constant')
        density_map[category, :, :] += density

    return density_map


class Resize(object):
    def __init__(self, min_size=800, max_size=1333):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return oh, ow

    def __call__(self, size):
        size = self.get_size(size)
        return size


class COCODensityDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
            self, ann_file, root, remove_images_without_annotations, use_density_map=True, transforms=None
    ):
        super(COCODensityDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.use_density_map = use_density_map
        self.density_map_size = 100
        self.ids = sorted(self.ids)
        self.density_categories = 1
        self.density_map_stride = 1.0 / 8
        self.density_min_sigma = 1.0

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def get_annotation(self, image_id):
        coco = self.coco
        ann_ids = coco.getAnnIds(imgIds=image_id)
        img_data = self.coco.imgs[image_id]
        anno = coco.loadAnns(ann_ids)
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, (img_data['width'], img_data['height']), mode="xywh").convert("xyxy")

        labels = [obj["category_id"] for obj in anno]
        labels = [self.json_category_id_to_contiguous_id[c] for c in labels]
        target.add_field("labels", torch.tensor(labels))

        target = target.clip_to_image(remove_empty=True)

        return {'boxes': target.bbox.tolist(), 'labels': target.get_field('labels').tolist()}

    def __getitem__(self, idx):
        img, anno = super(COCODensityDataset, self).__getitem__(idx)
        width, height = img.size[0], img.size[1]
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        labels = classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        target = target.clip_to_image(remove_empty=True)

        if self.use_density_map:
            resize = Resize()
            input_height, input_width = resize((width, height))
            stride = self.density_map_stride
            output_height, output_width = round(input_height * stride), round(input_width * stride)
            size = (output_height, output_width)
            scale = (output_height / height, output_width / width)
            super_categories = [contiguous_coco_category_to_super_category(category, self.density_categories) for category in labels]
            density_map = generate_density_map(super_categories, target.bbox.tolist(), scale=scale, size=size,
                                               num_classes=self.density_categories, min_sigma=self.density_min_sigma)
            target.add_field('heatmap', Heatmap(torch.from_numpy(density_map)))

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


class CocoUnlabelDataset(Dataset):
    def __init__(self, img_dir, ann_file, pseudo_labels_file=None, use_density_map=True, transforms=None):
        """
        Args:
            img_dir:
            ann_file: dict_keys(['info', 'images', 'licenses'])
                {'license': 2, 'file_name': '000000533083.jpg', 'coco_url': 'http://images.cocodataset.org/unlabeled2017/000000533083.jpg', 'height': 640, 'width': 426, 'date_captured': '2013-11-14 10:56:14',
                'flickr_url': 'http://farm3.staticflickr.com/2567/4077404434_1bdea2d393_z.jpg', 'id': 533083}
        """
        from pycocotools.coco import COCO
        self.img_dir = img_dir
        self.use_density_map = use_density_map
        self.transforms = transforms
        self.coco = COCO('/data7/lufficc/coco/annotations/instances_minival2014.json')
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        with open(ann_file) as f:
            annotations = json.load(f)
            self.images = annotations['images']

        self.images_dict = {}
        for i in self.images:
            self.images_dict[i['id']] = i

        if pseudo_labels_file is None:
            self.annotations = None
        else:
            self.images = []
            with open(pseudo_labels_file) as fid:
                annotations = {}
                anns = json.load(fid)
                for ann in anns:
                    if len(ann['bbox']) > 0:
                        img_id = ann['id']
                        img_info = self.images_dict[img_id]
                        self.images.append(img_info)
                        annotations[img_id] = ann
                self.annotations = annotations
                print('Pseudo labels:  ', len(self.annotations))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_info = self.images[index]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        width, height = img.size[0], img.size[1]
        if self.annotations is None:
            target = BoxList(torch.empty((0, 4), dtype=torch.float32), (width, height), mode="xyxy")
        else:
            boxes = []
            labels = []
            ann = self.annotations[img_info['id']]
            for category, x, y, w, h in ann['bbox']:
                boxes.append([x, y, x + w, y + h])
                labels.append(category)

            target = BoxList(torch.tensor(boxes, dtype=torch.float32).reshape((-1, 4)), (width, height), mode="xyxy")
            target.add_field('labels', torch.tensor(labels))
            target = target.clip_to_image(remove_empty=True)

            if self.use_density_map:
                resize = Resize()
                input_height, input_width = resize((width, height))
                stride = 1.0 / 8
                output_height, output_width = round(input_height * stride), round(input_width * stride)
                size = (output_height, output_width)
                scale = (output_height / height, output_width / width)
                num_classes = 1
                super_categories = [0] * len(labels)
                density_map = generate_density_map(super_categories, target.bbox.tolist(), scale=scale, size=size, num_classes=num_classes)
                target.add_field('heatmap', Heatmap(torch.from_numpy(density_map)))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def get_img_info(self, index):
        img_info = self.images[index]
        return img_info
