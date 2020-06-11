This repository is dedicated to an assigment to identify faults in parts of equipmemnt captured in images. 

Task - 
--------

Good image             |  Bad image
:-------------------------:|:-------------------------:
![good image](assets/good_image.png)  |  ![bad image](assets/bad_image.png)

The image above on the left is an image of a metallic part without any defects and image above on the left is a sample with a bad part (nut missing). There are going to be different kinds of defects (like rust,dent etc.) present anywhere on the part.

The assignment is to come up with a deep-learning based solution that can differentiate b/w good and bad parts and visualise the location of the defect. The solution should take into account that new types of defects are often discovered in the field and should be designed to incorporate new defects with as few changes as possible.

Pipeline - 
------------

1) Data augmentation -  
	For both good and bad image.

2) Classification Neural net -
	Siamese nural net (VGG model with some changes to the final 2 layers).
	* VGG net : 4096 features  
	Look at modelsummary to get the structure. 
	The neural net identifies whther the two images are similar or different

3) Segmentation neural net -
	* For now, rather than using a neural net, we go with SIFT image keypoint matching

	This net is used to identify keypoints in the two images deemed different based on the classification neural net.
	These keypoints are matched to each other. The remaining unmatched keypoints in the tested are segmented.  

4) Visualize data 


	
Other possibilies - 
---------------------

* Siamese neural network - other models to be tested - MobileNet, Resnet,Inception, Xception
