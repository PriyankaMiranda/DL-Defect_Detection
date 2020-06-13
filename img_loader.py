import os
import numpy as np
import pickle

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input


train_folders= ['assets/good_images/', 'assets/bad_images/']
val_folders  = ['assets/good_val_data/', 'assets/bad_val_data/']
save_path = 'assets/'

classes = {
  "bad": 0,
  "good": 1
}

def load_images(folder_paths):
    X = []
    y = []
    category_images=[]
    for folder in folder_paths:
        className = folder.split("/")[1].split("_")[0]
        for imgs in os.listdir(folder):
            image_path = os.path.join(folder, imgs)
            
            # load an image from file
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)

            category_images.append(image)
            y.append(classes[className])
            X.append(image)
    X = np.stack(X)
    y = np.vstack(y)
    return X, y

X, y = load_images(train_folders)
with open(os.path.join(save_path,"train.pickle"), "wb") as f:
    pickle.dump((X,y),f)

Xval,yval=load_images(val_folders)
with open(os.path.join(save_path,"val.pickle"), "wb") as f:
    pickle.dump((Xval,yval),f)