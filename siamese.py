from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16

from keras import backend as K
from keras.optimizers import Adam

from keras.layers.core import Lambda, Flatten, Dense
from keras.layers import Input
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import GlobalMaxPool2D
import numpy as np

from train import Train_Class  

def initialize_bias(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def generate_model():

    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))
    
    base_model = VGG16()

    for x in base_model.layers[:-1]:
        x.trainable = True

    encoded_l = base_model(input_1)
    encoded_r = base_model(input_2)
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    siamese_net = Model(inputs=[input_1,input_2],outputs=prediction)

    return siamese_net

if __name__ == "__main__":
    model = generate_model()    
    model.summary() 
    optimizer = Adam(lr = 0.00006)
    model.compile(loss="binary_crossentropy",optimizer=optimizer)
    
    print("Model loaded!")

    Train_Class.train(Train_Class, model)
































