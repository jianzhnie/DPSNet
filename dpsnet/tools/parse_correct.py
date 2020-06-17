import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="DPNet Parse Correct")
    parser.add_argument(
        "--pseudo_label",
        required=True,
        metavar="FILE",
        help="path to pseudo file",
        type=str,
    )
    parser.add_argument(
        "--ann_file",
        default='/data7/lufficc/rpc/instances_test2019.json',
        metavar="FILE",
        help="path to gt annotation file",
        type=str,
    )

    args = parser.parse_args()

    with open(args.ann_file) as fid:
        gt_annotations = json.load(fid)

    annotations = defaultdict(list)
    images = []
    for image in gt_annotations['images']:
        images.append(image)
    for ann in gt_annotations['annotations']:
        bbox = ann['bbox']
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        annotations[ann['image_id']].append((ann['category_id'], x, y, w, h))
    del gt_annotations

    with open(args.pseudo_label) as fid:
        pseudo_annotation = json.load(fid)

    correct = 0
    for ann in pseudo_annotation:
        pseudo_labels = [item[0] for item in ann['bbox']]

        gt_labels = [item[0] for item in annotations[ann['id']]]

        if sorted(pseudo_labels) == sorted(gt_labels):
            correct += 1
    print('Ratio: {:.3f} ({}/{})'.format(correct / len(pseudo_annotation), correct, len(pseudo_annotation)))


if __name__ == "__main__":
    main()
