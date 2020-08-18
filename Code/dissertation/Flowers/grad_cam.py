# -*- coding: utf-8 -*-
"""grad_cam

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/grad_cam.ipynb

# Grad-CAM class activation visualization

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2020/04/26<br>
**Last modified:** 2020/05/14<br>
**Description:** How to obtain a class activation heatmap for an image classification model.

Adapted from Deep Learning with Python (2017).

## Setup
"""
import numpy as np
import tensorflow as tf
from ipython_genutils.py3compat import xrange
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""## Configurable parameters

You can change these to another model.

To get the values for `last_conv_layer_name` and `classifier_layer_names`, use
 `model.summary()` to see the names of all layers in the model.
"""

model_builder = keras.applications.xception.Xception
img_size = (250, 250)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

# last_conv_layer_name = "conv2d_2"
# classifier_layer_names = [
#     "max_pooling2d_2",
#     "flatten",
# ]


last_conv_layer_name = "conv2d"
classifier_layer_names = ["global_average_pooling2d", "dense_1", ]

# The local path to our target image
# img_path = keras.utils.get_file(
#     "african_elephant.jpg", " https://i.imgur.com/Bvro0YD.png"
# )

# display(Image(img_path))

"""## The Grad-CAM algorithm"""

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def shrink_shap(data, rows, cols):
    shrunk = np.zeros((rows,cols))
    for i in xrange(0,rows):
        for j in xrange(0,cols):
            row_sp = int (data.shape[0]/rows)
            col_sp = int (data.shape[1]/cols)
            zz = data[i*row_sp : i*row_sp + row_sp, j*col_sp : j*col_sp + col_sp]
            shrunk[i,j] = np.sum(zz)
    values =  np.array([(item) for sublist in shrunk/np.max(shrunk) for item in sublist])
    shap_values = []
    for i in range(5):
        shap_values.append([[item for sublist in values for item in sublist]])
    shap_values = np.array(shap_values)
    shap_values = shap_values / np.max(shap_values)

def test_drive_grad(img_array):
    """## Let's test-drive it"""

    # Prepare image
    # img_array = preprocess_input(get_img_array(img_path, size=img_size))

    # Make model
    model = tf.keras.models.load_model('flowers-vgg')

    # Print what the top predicted class is
    # preds = model.predict(img_array.reshape(1,250, 250, 3))
    # print("Predicted:", preds)

    # last_conv_layer_name = "conv2d"
    # classifier_layer_names = ["global_average_pooling2d", "dense_1", ]
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(
        img_array.reshape(1,250, 250, 3), model, last_conv_layer_name, classifier_layer_names
    )

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()
    """## Create a superimposed visualization"""

    # We load the original image
    # img = keras.preprocessing.image.load_img(img_path)
    # img = keras.preprocessing.image.img_to_array(img)
    img = img_array

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("binary")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    print(heatmap.shape[1] + heatmap.shape[0])
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.015 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    save_path = "elephant_cam.jpg"
    superimposed_img.save(save_path)

    plt.imshow(superimposed_img)
    plt.show()
    plt.close()
    # Display Grad CAM
    # display(Image(save_path))