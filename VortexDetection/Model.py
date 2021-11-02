#####YOLOv3 architecture  #######

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from VortexDetection.Dataset import dataset
from VortexDetection import Configration

############################## Build Residual Block ##########################################
class Res_Block(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNN(channels, channels // 2, kernel_size=1),
                    CNN(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, k):
        for layer in self.layers:
            if self.use_residual:
                k = k + layer(k)
            else:
                k = layer(k)
        return k

class CNN(nn.Module):
    def __init__(self, input_channels, output_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, k):
        if self.use_bn_act:
            return self.leaky_relu(self.bn(self.conv(k)))
        else:
            return self.conv(k)

class Scale_Prediction(nn.Module):
    def __init__(self, input_channels, no_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNN(input_channels, 2 * input_channels, kernel_size=3, padding=1),
            CNN(
                2 * input_channels, (no_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.no_classes = no_classes

    def forward(self, k):
        return (
            self.pred(k)
            .reshape(k.shape[0], 3, self.no_classes + 5, k.shape[2], k.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

class CV_DetectionVorticies_YOLOv3(nn.Module):
    def __init__(self, input_channels=3, no_classes=2):
        super().__init__()
        self.input_channels = input_channels
        self.no_classes = no_classes
        self.layers = self._create_conv_layers()

    def forward(self, k):
        outputOfEachResolution = []  # for each resolution
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, Scale_Prediction):
                outputOfEachResolution.append(layer(k))
                continue
            k = layer(k)

            if isinstance(layer, Res_Block) and layer.num_repeats == 8:
                route_connections.append(k)

            elif isinstance(layer, nn.Upsample):
                k = torch.cat([k, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputOfEachResolution

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        input_channels = self.input_channels

        for module_cofig in Configration.configration:
            if isinstance(module_cofig, tuple):
                output_channels, kernel_size, stride = module_cofig
                layers.append(
                    CNN(input_channels, output_channels, kernel_size=kernel_size,
                        stride=stride, padding=1 if kernel_size == 3 else 0,))
                input_channels = output_channels

            elif isinstance(module_cofig, list):
                num_repeats = module_cofig[1]
                layers.append(Res_Block(input_channels, num_repeats=num_repeats,))

            elif isinstance(module_cofig, str):
                if module_cofig == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    input_channels = input_channels * 3

                elif module_cofig == "S":
                    layers += [
                        Res_Block(input_channels, use_residual=False, num_repeats=1),
                        CNN(input_channels, input_channels // 2, kernel_size=1),
                        Scale_Prediction(input_channels // 2, no_classes=self.no_classes),
                    ]
                    input_channels = input_channels // 2

        return layers

##### get loaders of the image ###################################################
def loaders(img_path):
    test_dataset = dataset(
        img_path,
        anchors=Configration.ANCHORS,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
      )
    return test_loader

####################### Load the checkpoint ##############################################
def load_checkpoint(checkpoint, model, optimizer):
    print("==> Loading please wait!")
    get_checkpoint = torch.load(checkpoint, map_location=Configration.DEVICE)
    model.load_state_dict(get_checkpoint["state_dict"])
    optimizer.load_state_dict(get_checkpoint["optimizer"])

def b1(box):
    return box[1]
def non_max_suppression( iou_threshold, threshold, pred_boxes, box_format="corners"):
    """
    The Parameters of Non Max Suppression is:
        iou_threshold (float): threshold where predicted pred_boxes is correct
        threshold (float): threshold to remove predicted pred_boxes (independent of IoU)
        pred_boxes (list): list of lists containing all pred_boxes with each pred_boxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        box_format (str): "midpoint" or "corners" used to specify pred_boxes
    Returns:
        list: pred_boxes after applying NMS given a specific IoU threshold
    """

    boxesAppend=[]
    for box in pred_boxes:
        if box[1] > threshold:
            boxesAppend.append(box)
    pred_boxes = boxesAppend
    ######### Sort the prediction boxes order by probability score #####################
    pred_boxes = sorted(pred_boxes, key=lambda d: d[1], reverse=True)
    nms_boxes = []
    ######## suppressed the predicting boxes which IOU < iou_threshold #########
    while pred_boxes:
        box_perdI = pred_boxes.pop(0)
        pred_boxes = [
            box
            for box in pred_boxes
            if box[0] != box_perdI[0]
            or intersection_over_union(
                torch.tensor(box_perdI[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        nms_boxes.append(box_perdI)

    return nms_boxes

def intersection_over_union(box_perdI, comp_boxes, box_format="midpoint"):
    """
    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        box_perdI (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        comp_boxes (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all prediction boxes
    """
    if box_format == "midpoint":
        box1_x1 = box_perdI[..., 0:1] - box_perdI[..., 2:3] / 2
        box1_y1 = box_perdI[..., 1:2] - box_perdI[..., 3:4] / 2
        box1_x2 = box_perdI[..., 0:1] + box_perdI[..., 2:3] / 2
        box1_y2 = box_perdI[..., 1:2] + box_perdI[..., 3:4] / 2
        box2_x1 = comp_boxes[..., 0:1] - comp_boxes[..., 2:3] / 2
        box2_y1 = comp_boxes[..., 1:2] - comp_boxes[..., 3:4] / 2
        box2_x2 = comp_boxes[..., 0:1] + comp_boxes[..., 2:3] / 2
        box2_y2 = comp_boxes[..., 1:2] + comp_boxes[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

