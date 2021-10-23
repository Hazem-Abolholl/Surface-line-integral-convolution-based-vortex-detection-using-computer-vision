# Surface line integral convolution-based vortex detection using computer vision
**Version 1.0.0**

This is your README. READMEs are where you can communicate what your project is and how to use it.

We proposed a new approach using convolutional neural networks to detect flow structures directly from streamline plots, using the line integral convolution method. We show that our computer vision-based approach is able to reduce the number of false positives and negatives entirely.
Write your name on line 6, save it, and then head back to GitHub Desktop.

## Requirements
The Vortex Detection using Computer Vision based on YOLOv3 works only on Python 3.7 and superior. The following library are used:
*	Torchvision
*	PyTorch/1.6.0-
*	Albumentations
*	config
*	NumPy
*	Matplotlib
*	Tensorboard

## Tensorboard
Track training progress in:
1. Mean Average Precision (mAP value)
2. Loss Function
3. Class accuracy - Object accuracy - No Object accuracy

For mAP and Loss function:
`tensorboard --logdir=logs`

For correct_class, correct_obj, and correct_Noobj:
`tensorboard --logdir=runs`

## Download pretrained weights

## Results
<table align="center" style="border: 0"> 
  <tr>
		<td><img src="testimage1.png" height="250" width="250" style="border: 0">    
    </td>
    <td><img src="testimage2.png" height="250" width="250" style="border: 0">    
    </td>

 </tr>
	<tr align="center" >
	<td><center>Test image1</center></td>
    <td><center>Test image2</center></td>

  </tr>
  <tr align="center">
    <td colspan="2" >Fig.1 - The output result of test images. The red box is the predicted bounding box</td>
  </tr>	
 </table>


