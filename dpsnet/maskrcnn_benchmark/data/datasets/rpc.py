import glob
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data
from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import Heatmap
from maskrcnn_benchmark.utils.density import generate_density_map, rpc_category_to_super_category

DENSITY_MAP_WIDTH = 100
DENSITY_MAP_HEIGHT = 100


# --------------------------------------------
# ----------------Test dataset----------------
# --------------------------------------------
class RPCTestDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, ann_file, transforms=None):
        self.transforms = transforms
        self.images_dir = images_dir
        self.ann_file = ann_file

        with open(self.ann_file) as fid:
            data = json.load(fid)

        annotations = defaultdict(list)
        images = []
        for image in data['images']:
            images.append(image)
        for ann in data['annotations']:
            bbox = ann['bbox']
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            annotations[ann['image_id']].append((ann['category_id'], x, y, w, h))

        self.images = images
        self.annotations = dict(annotations)

    def __getitem__(self, index):
        image_id = self.images[index]['id']
        img_path = os.path.join(self.images_dir, self.images[index]['file_name'])
        img = Image.open(img_path).convert("RGB")
        width, height = img.size[0], img.size[1]
        boxes = []
        labels = []
        ann = self.annotations[image_id]
        for category, x, y, w, h in ann:
            boxes.append([x, y, x + w, y + h])
            labels.append(category)

        target = BoxList(torch.tensor(boxes, dtype=torch.float32), (width, height), mode="xyxy")
        target.add_field('labels', torch.tensor(labels))
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def get_annotation(self, image_id):
        ann = self.annotations[image_id]
        return ann

    def __len__(self):
        return len(self.images)

    def get_img_info(self, index):
        image = self.images[index]
        return {"height": image['height'], "width": image['width'], "id": image['id'], 'file_name': image['file_name']}


# --------------------------------------------
# ----------------Train dataset---------------
# --------------------------------------------
class RPCDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images_dir,
                 ann_file,
                 use_density_map=False,
                 rendered=False,
                 transforms=None):
        self.images_dir = images_dir
        self.ann_file = ann_file
        self.use_density_map = use_density_map
        self.rendered = rendered
        self.transforms = transforms
        self.density_categories = 1
        self.density_map_stride = 1.0 / 8
        self.density_min_sigma = 1.0

        self.scale = 1.0
        self.ext = '.jpg'
        self.image_size = 1815

        if self.rendered:  # Rendered image is 800*800 and format is png
            self.scale = 800.0 / 1815.0
            #self.ext = '.png'
            self.image_size = 800

        with open(self.ann_file) as fid:
            self.annotations = json.load(fid)

    def __getitem__(self, index):
        ann = self.annotations[index]
        image_id = ann['image_id']
        image_name = os.path.splitext(image_id)[0]
        img_path = os.path.join(self.images_dir, image_name + self.ext)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size[0], img.size[1]
        boxes = []
        labels = []
        objects = ann['objects']
        for item in objects:
            category = item['category_id']
            x, y, w, h = item['bbox']
            boxes.append([x * self.scale, y * self.scale, (x + w) * self.scale, (y + h) * self.scale])
            labels.append(category)

        target = BoxList(torch.tensor(boxes, dtype=torch.float32), (width, height), mode="xyxy")
        target.add_field('labels', torch.tensor(labels))

        if self.use_density_map:
            image_size = self.image_size
            size = int(self.density_map_stride * 800)
            num_classes = self.density_categories
            assert img.width == image_size
            assert img.height == image_size
            super_categories = [rpc_category_to_super_category(category, num_classes) for category in labels]
            density_map = generate_density_map(super_categories, boxes,
                                               scale=size / image_size,
                                               size=size, num_classes=num_classes,
                                               min_sigma=self.density_min_sigma)
            target.add_field('heatmap', Heatmap(torch.from_numpy(density_map)))

        target = target.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.annotations)

    def get_img_info(self, index):
        image_size = 800 if self.rendered else 1815
        return {"height": image_size, "width": image_size}


class RPCPseudoDataset(torch.utils.data.Dataset):

    def __init__(self, images_dir, ann_file=None, use_density_map=False, annotations=None, transforms=None):
        self.images_dir = images_dir
        self.ann_file = ann_file
        self.use_density_map = use_density_map
        self.transforms = transforms
        self.density_categories = 1
        self.density_map_stride = 1.0 / 8
        self.density_min_sigma = 1.0

        if annotations is not None:
            self.annotations = annotations
        else:
            with open(self.ann_file) as fid:
                annotations = json.load(fid)
            self.annotations = annotations

        print('Valid annotations: {}'.format(len(self.annotations)))

    def __getitem__(self, index):
        ann = self.annotations[index]
        img_path = os.path.join(self.images_dir, ann['file_name'])
        img = Image.open(img_path).convert("RGB")
        width, height = img.size[0], img.size[1]
        boxes = []
        labels = []
        for category, x, y, w, h in ann['bbox']:
            boxes.append([x, y, x + w, y + h])
            labels.append(category)

        target = BoxList(torch.tensor(boxes, dtype=torch.float32), (width, height), mode="xyxy")
        target.add_field('labels', torch.tensor(labels))
        target = target.clip_to_image(remove_empty=True)
        if self.use_density_map:
            size = int(800 * self.density_map_stride)
            image_size = img.width  # Test images are squares, except 20180824-14-36-38-430.jpg(1860x1859)
            num_classes = self.density_categories
            super_categories = [rpc_category_to_super_category(category, self.density_categories) for category in labels]
            density_map = generate_density_map(super_categories, boxes,
                                               scale=size / image_size,
                                               size=size,
                                               num_classes=num_classes,
                                               min_sigma=self.density_min_sigma)
            target.add_field('heatmap', Heatmap(torch.from_numpy(density_map)))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.annotations)

    def get_img_info(self, index):
        ann = self.annotations[index]
        return {"height": ann['height'], "width": ann['width'], "id": ann['id'], 'file_name': ann['file_name']}


class RPCInstanceSelectDataset(torch.utils.data.Dataset):

    def __init__(self, images_dir, ann_file, transforms=None):
        self.images_dir = images_dir
        self.ann_file = ann_file
        self.transforms = transforms
        self.images_dir = images_dir
        self.threshold = 0.95

        with open(self.ann_file) as fid:
            annotations = json.load(fid)

        delete_keys = []
        total_objects = 0
        filtered_objects = 0
        annotation_dict = defaultdict(list)
        for annotation in annotations:
            annotation_dict[annotation['image_id']].append(annotation)
        for image_id in annotation_dict:
            count = 0
            for obj in annotation_dict[image_id]:
                total_objects += 1
                if obj['score'] > self.threshold:
                    filtered_objects += 1
                    count += 1
            if count == 0:
                delete_keys.append(image_id)

        with open('/data/wedward/RPC_dataset/instances_test2019.json') as fid:
            data = json.load(fid)

        images = []
        for image in data['images']:
            if image['id'] not in delete_keys:
                images.append(image)

        for image_id in delete_keys:
            del annotation_dict[image_id]

        self.annotations = dict(annotation_dict)
        self.images = images
        assert len(self.images) == len(self.annotations)

        print('Valid annotations: {}'.format(len(self.annotations)))
        print('Ratio: {:.3f}({}/{})'.format(filtered_objects / total_objects, filtered_objects, total_objects))

    def __getitem__(self, index):
        ann = self.annotations[self.images[index]['id']]
        img_path = os.path.join(self.images_dir, self.images[index]['file_name'])
        img = Image.open(img_path).convert("RGB")
        width, height = img.size[0], img.size[1]
        boxes = []
        labels = []
        viz = False
        for obj in ann:
            if obj['score'] > self.threshold:
                category = obj['category_id']
                x, y, w, h = obj['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(category)
            else:
                x, y, w, h = [round(k) for k in obj['bbox']]
                img = np.array(img)
                img[y:y + h, x:x + w, :] = (164, 166, 164)
                img = Image.fromarray(img, mode='RGB')
        if viz:
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.show()
            quit()

        target = BoxList(torch.tensor(boxes, dtype=torch.float32), (width, height), mode="xyxy")
        target.add_field('labels', torch.tensor(labels))
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.images)

    def get_img_info(self, index):
        ann = self.images[index]
        return ann


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        self.folder = '/data/wedward/RPC_dataset/train2019/'
        self.paths = glob.glob(os.path.join(self.folder, '*.jpg'))
        random.shuffle(self.paths)
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        width, height = img.size[0], img.size[1]
        boxes = np.zeros([0, 4], dtype=np.float32)
        target = BoxList(torch.tensor(boxes, dtype=torch.float32), (width, height), mode="xyxy")
        if self.transforms:
            img, _ = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.paths)
