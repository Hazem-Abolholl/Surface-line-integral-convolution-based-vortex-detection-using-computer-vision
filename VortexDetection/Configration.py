import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Image_Size =416
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],[(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # these have been Normalized to be between [0, 1]
CHECKPOINT = "Weight/weights.pth.tar"
R = [Image_Size // 32, Image_Size // 16, Image_Size // 8]
###################################################################################################
####################architecture config according to the YOLOv3 paper:#############################
configration = [
   ## Darknet-53 ##################################################################################
    (32, 3, 1), (64, 3, 2),   # Tuple is knowing as(filters, kernel_size, stride)                 #
    ["B", 1],                 # "B" indicating a residual block followed by the number of repeats #
    (128, 3, 2),["B", 2],(256, 3, 2),["B", 8],(512, 3, 2),["B", 8], (1024, 3, 2),                 #
    ["B", 4], (512, 1, 1), (1024, 3, 1),                                                          #
   ################################################################################################
    "S",             # "S" is for scale prediction block and computing the yolo loss
    (256, 1, 1),"U", # "U" is for upsampling the feature map and concatenating with a previous layer
    (256, 1, 1), (512, 3, 1),"S",(128, 1, 1), "U", (128, 1, 1), (256, 3, 1), "S",
]
