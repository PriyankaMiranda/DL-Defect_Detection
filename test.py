import os
import time 
import numpy as np
import pickle

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
	

def simple_test():
	model_loc = ""
	loaded_model = pickle.load(open(model_loc, 'rb'))
	test_img_loc = ""
	test_loc_class = 0 or 1

	image = load_img(test_img_loc, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	y.append(test_loc_class)
	X.append(image)
	X = np.stack(X)
	y = np.vstack(y)
	result = loaded_model.score(X, y)
	print(result)
	

if __name__ == "__main__":
	simple_test()
