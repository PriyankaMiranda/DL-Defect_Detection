import cv2 
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models


# alexnet_model = models.alexnet(pretrained=True)
resnet18 = models.resnet18(pretrained=True)

#reading image
img1 = cv2.imread('assets/good_image.png')  

#reading image
# img2 = cv2.imread('assets/bad_val_data/image_0_9404.jpg')  
img2 = cv2.imread('assets/bad_image.png')  

out = model.predict_segmentation(
    inp="assets/good_image.png",
    out_fname="output.png"
)

print(img1)
print(img1.shape)
print(out)
print(out.shape)
input()


# plt.imshow(img1)
# plt.show()
# plt.imshow(img2)
# plt.show()
