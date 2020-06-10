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

2) Data augmentation -  
	For both good and bad image.

2) Feature extraction - 
	Remove last softmax layer and get features for this image from existing pre-trained model (VGG, MobileNet, Resnet..)
	Features extracted from models :
	* VGG net : 4096 features  

3) Training -
	To do ....
	Simple ML techniques

	
Other possibilies - 
---------------------

* Siamese neural networks
