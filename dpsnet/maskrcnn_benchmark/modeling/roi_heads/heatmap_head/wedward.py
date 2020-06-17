import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.num_classes = cfg.MODEL.DENSITY_HEAD.NUM_CLASSES
        self.level = cfg.MODEL.DENSITY_HEAD.FPN_LEVEL
        self.mode = cfg.MODEL.DENSITY_HEAD.INTERPOLATE_MODE

        self.heatmap_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1),
            nn.ReLU() #nn.Sigmoid() if change, there would be error like 
        )
        
        """
        INFO:maskrcnn_benchmark.trainer:eta: 1 day, 7:39:00  iter: 120  loss: nan (nan)  loss_box_reg: nan (nan)  loss_classifier: nan (nan)  loss_density: 0.8340 (161.0250)  loss_objectness: nan (nan)  loss_rpn_box_reg: nan (nan)  time: 0.6104 (0.6334)  data: 0.0108 (0.0318)  lr: 0.004920  max mem: 6252
WARNING:root:NaN or Inf found in input tensor.
        
        """
        self.convs_1x1 = nn.ModuleList()
        for i in range(4):
            self.convs_1x1.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            )
        self.density_criterion = nn.MSELoss(reduction='sum')
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.zeros_(m.bias)

    def forward(self, features, detections, targets=None):
        """
        Args:
            features: Feature pyramid from FPN: (p2, p3, p4, p5, p6)
            detections: list[Boxlist]
            targets: list[Boxlist]
        Returns:
            predication, detections, loss dict
        """
        if self.level == 1:
            return self.compat_forward(features, detections, targets)

        features = list(features)
        conv1x1_index = 0
        for i in range(0, self.level):
            feature = F.interpolate(features[i], size=features[i + 1].shape[2:], mode=self.mode, align_corners=False)
            feature = self.convs_1x1[conv1x1_index](feature)
            conv1x1_index += 1
            features[i + 1] = feature + features[i + 1]
        for i in range(len(features) - 1, self.level, -1):
            feature = F.interpolate(features[i], size=features[i - 1].shape[2:], mode=self.mode, align_corners=False)
            feature = self.convs_1x1[conv1x1_index](feature)
            conv1x1_index += 1
            features[i - 1] = feature + features[i - 1]

        feature = features[self.level]

        density_logits = self.heatmap_convs(feature)

        if not self.training:
            density = torch.sum(density_logits, dim=(2, 3))  # (batch, num_classes)
            for i, detection in enumerate(detections):
                detection.add_field('density', density[i])
            return feature, density_logits, detections, {}

        gt_density = []
        for i, target in enumerate(targets):
            gt_heatmap = target.get_field('heatmap').heatmap
            if tuple(gt_heatmap.shape) != tuple(density_logits[i].shape):
                gt_heatmap = F.pad(gt_heatmap, [0, density_logits[i].shape[2] - gt_heatmap.shape[2],
                                                0, density_logits[i].shape[1] - gt_heatmap.shape[1]])
            gt_density.append(gt_heatmap)

        gt_density = torch.stack(gt_density, dim=0)
        loss_density = self.density_criterion(density_logits, gt_density)
        return feature, density_logits, detections, dict(loss_density=loss_density)

    def compat_forward(self, features, detections, targets=None):
        """
        Args:
            features: Feature pyramid from FPN: (p2, p3, p4, p5, p6)
            detections: list[Boxlist]
            targets: list[Boxlist]
        Returns:
            predication, detections, loss dict
        """
        p2, p3, p4, p5, p6 = features
        mode = 'bilinear'
        p6 = F.interpolate(p6, size=p5.shape[2:], mode=mode, align_corners=False)
        p5 = p5 + self.convs_1x1[0](p6)

        p5 = F.interpolate(p5, size=p4.shape[2:], mode=mode, align_corners=False)
        p4 = p4 + self.convs_1x1[1](p5)

        p4 = F.interpolate(p4, size=p3.shape[2:], mode=mode, align_corners=False)
        p2 = F.interpolate(p2, size=p3.shape[2:], mode=mode, align_corners=False)

        feature = self.convs_1x1[2](p4) + self.convs_1x1[3](p2) + p3

        density_logits = self.heatmap_convs(feature)

        if not self.training:
            density = torch.sum(density_logits, dim=(2, 3))  # (batch, num_classes)
            for i, detection in enumerate(detections):
                detection.add_field('density', density[i])
                # temp add
                detection.add_field('density_map', density_logits.cpu())
            return feature, density_logits, detections, {}

        gt_density = []
        for i, target in enumerate(targets):
            gt_heatmap = target.get_field('heatmap').heatmap
            if tuple(gt_heatmap.shape) != tuple(density_logits[i].shape):
                gt_heatmap = F.pad(gt_heatmap, [0, density_logits[i].shape[2] - gt_heatmap.shape[2],
                                                0, density_logits[i].shape[1] - gt_heatmap.shape[1]])
            gt_density.append(gt_heatmap)

        gt_density = torch.stack(gt_density, dim=0)
        loss_density = self.density_criterion(density_logits, gt_density)
        return feature, density_logits, detections, dict(loss_density=loss_density)


def build_heatmap_head(cfg, in_channels):
    return HeatmapHead(cfg, in_channels=in_channels)
