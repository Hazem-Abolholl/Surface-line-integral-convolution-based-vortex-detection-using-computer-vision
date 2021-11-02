import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from VortexDetection.Model import(non_max_suppression)

def plot_image(model, loader, thresh, iou_thresh, anchors):
    model.eval()
    x, y = next(iter(loader))
    x = x.to("cpu")
    with torch.no_grad():
        out = model(x)
        pred_boxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, R, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = prepare_Ped_boxes(
                out[i], anchor, R=R, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                pred_boxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
             iou_thresh, thresh, pred_boxes[i],box_format="midpoint",
        )
    image = x[0].permute(1,2,0).detach().cpu()
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    count = 0
    for box in nms_boxes:
        count += 1
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor='r',
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.title(f'Number of vortices is: {count}')
    plt.show()


def prepare_Ped_boxes(predictions, anchors, R, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image
    argument:
    predictions: tensor of size (N, 3, S, S, 7)
    anchors: the anchors used for the predictions
    R: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    return:
    converted_pred_boxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]


    cell_indices = (
        torch.arange(R)
        .repeat(predictions.shape[0], 3, R, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / R * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / R * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / R * box_predictions[..., 2:4]
    converted_pred_boxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * R * R, 6)
    return converted_pred_boxes.tolist()






