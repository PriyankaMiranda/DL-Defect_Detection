# from keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace

from keras.models import Model

from keras import layers
from keras.layers import GlobalMaxPool2D , GlobalAvgPool2D, Input, Dense, Concatenate, Multiply, Dropout, Subtract
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers.merge import Concatenate

from keras.optimizers import Adam

from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import numpy as np
import io

from train import Train_Class  

def initialize_weights(shape, name=None, dtype=None):
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, dtype=None):
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def sum_fn(x):
    return K.sum(x, axis=1, keepdims=True)


def sum_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1,1)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)


def generate_model():

    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    base_model = VGGFace(model='resnet50', include_top=False)

    for x in base_model.layers[:-3]:
        x.trainable = True

    x1 = base_model(input_1)
    x2 = base_model(input_2)

    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([x1, x1])
    x2_ = Multiply()([x2, x2])
    x4 = Subtract()([x1_, x2_])



    #https://stackoverflow.com/a/51003359/10650182
    x5 = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([x1, x2])

    x6 = Lambda(sum_fn, output_shape=sum_output_shape)(x3)

    x7 = Lambda(sum_fn, output_shape=sum_output_shape)(x4)


    x = Concatenate(axis=-1)([x7, x6, x5,x4, x3])

    x = Dense(200, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)



    return model

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

def myprint(s):
    with open('model_summary_day2_t5.txt','a') as f:
        print(s, file=f)


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

if __name__ == "__main__":
    model = generate_model()    
    model.summary()

    model_summary_string = get_model_summary(model)
    myprint(model_summary_string)

    # optimizer = Adam(lr = 0.00006)
    # model.compile(loss="binary_crossentropy",optimizer=optimizer)
    model.compile(loss=[focal_loss(alpha=.25, gamma=2)], metrics=['acc',auroc], optimizer=Adam(0.00001))
    print("Model loaded!")

    Train_Class.train(Train_Class, model)
































