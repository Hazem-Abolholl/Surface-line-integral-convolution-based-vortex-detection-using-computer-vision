import torch
import torch.optim as optim
import argparse
from VortexDetection import Configration
from VortexDetection.Model import (CV_DetectionVorticies_YOLOv3,loaders,
load_checkpoint)
from VortexDetection.PlotResult import (plot_image)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', dest='input_filename',
                        default='Data/testImage1.jpg', type=str,
                        )
    args = parser.parse_args()
    # Build the architecture of the model based on YOLOv3 according to the paper
    model = CV_DetectionVorticies_YOLOv3(no_classes=2).to(Configration.DEVICE)
    optimizer = optim.Adam(model.parameters())
    ## getting the loaders of specific test image
    loader = loaders(img_path=args.input_filename)
    # load the checkpoint after training the algorithm
    load_checkpoint(Configration.CHECKPOINT, model, optimizer)

    scaled_anchors = (torch.tensor(Configration.ANCHORS)
            * torch.tensor(Configration.R).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(Configration.DEVICE)
    #plot the test image
    plot_image(model, loader, 0.6, 0.5, scaled_anchors)

