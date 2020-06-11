from keras.models import Sequential
from keras.layers import 	Dense
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
# load the model


vgg16_model = VGG16()
model = Sequential()

for layer in vgg16_model.layers[:-1]: # this is where I changed your code
    model.add(layer)    

# Freeze the layers 
for layer in model.layers:
    layer.trainable = False

print(model.summary())
print(model.summary())

# load an image from file
image = load_img('assets/good_image.png', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
features = model.predict(image)
# convert the probabilities to class labels
