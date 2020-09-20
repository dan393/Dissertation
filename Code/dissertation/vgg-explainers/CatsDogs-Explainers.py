#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import tensorboard as tensorboard
import seaborn as seaborn
from tensorflow.python.client import device_lib
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
import os
import pandas as pd
import shap
import random
import numpy as np
import seaborn as sns
import csv
from skimage.util import random_noise
import time
import lime
from lime import lime_image
import sys
import grad_cam as grad
from importlib import reload
from skimage.segmentation import slic
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kr_helper_funcs as kr
print('tensorflow' + tf.__version__)
print('tensorboard' + tensorboard.__version__)
print('seaborn' + seaborn.__version__)
tf.config.list_physical_devices('GPU')
print(tf.test.is_built_with_cuda)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.list_physical_devices('GPU')
device_lib.list_local_devices()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))


# In[2]:


from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions


# In[3]:


data_dir="../../input/pets"
test_path= os.path.join(data_dir, 'test')

os.listdir(test_path)


# In[4]:


image_shape =(224, 224, 3)
test_gen = ImageDataGenerator(rescale =1./255)
batch_size=32
test_image_gen= test_gen.flow_from_directory(test_path, target_size=image_shape[:2], 
                                               color_mode='rgb', batch_size=batch_size, 
                                               class_mode='binary',  shuffle=False)
test_image_gen.class_indices


# In[5]:


import shap
import numpy as np
X_test, _ = test_image_gen.next()
# background = X_test[np.random.choice(20, 10, replace=False)]
background = X_test[0:5]
# explain predictions of the model on three images
# e = shap.DeepExplainer(tf.keras.models.load_model('flowers'), background)

model =VGG16()
# tulip_image_path = test_path + '/tulip/' + os.listdir(test_path + '/tulip/')[5]
# tulip_image_path = test_path + '/cats/' + os.listdir(test_path + '/cats/')[5]
tulip_image_path = 'sample.png'
# plt.imshow(imread(dog_image))
img = image.load_img(tulip_image_path, target_size=(224, 224, 3))
# plt.imshow(imread(tulip_image_path))
img_orig = image.img_to_array(img)
# standardized_image = test_gen.standardize(img_orig)
# plt.imshow((standardized_image))


# In[6]:


decode_predictions(model.predict(preprocess_input(np.expand_dims(img_orig.copy(), axis=0))))


# In[7]:


grad.test_drive_grad_original(img_orig, model)


# In[8]:


os.listdir(test_path + '/cats/')


# In[ ]:





# In[ ]:





# In[9]:


# segment the image so we don't have to explain every pixel
segments_slic = slic(img, n_segments=49, compactness=1000, sigma=3)

# define a function that depends on a binary mask representing if an image region is hidden
def mask_image(zs, segmentation, image, background=None):
    print((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    if background is None:
        background = image.mean((0,1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i,:,:,:] = image
        for j in range(zs.shape[1]):
            if zs[i,j] == 0:
                out[i][segmentation == j,:] = background
    return out
def f(z):
#     print("Call")
#     for i in z:
#         print (i)
#     print(img_orig.shape)

    return model.predict(preprocess_input(mask_image(z, segments_slic, img_orig.copy(), 255)))
#     return model.predict(mask_image(z, segments_slic, img_orig, 250))

# use Kernel SHAP to explain the network's predictions
explainer = shap.KernelExplainer(f, np.zeros((1,50)))
shap_values = explainer.shap_values(np.ones((1,50)) , nsamples=1000) # runs VGG16 1000 times


# In[10]:


# # use Kernel SHAP to explain the network's predictions
# explainer = shap.KernelExplainer(f, np.zeros((1,50)))
# shap_values = explainer.shap_values(np.ones((1,50)) , nsamples=1000) # runs VGG16 1000 times


# In[11]:


list(test_image_gen.class_indices.keys())


# In[12]:


# get the top predictions from the model
preds = model.predict(preprocess_input(np.expand_dims(img.copy(), axis=0)))
top_preds = np.argsort(-preds)


# In[13]:


decode_predictions(model.predict(preprocess_input(np.expand_dims(img.copy(), axis=0))))


# In[14]:


plt.imshow((np.expand_dims(img_orig.copy(), axis=0))[0])


# In[15]:


plt.imshow(img_orig.copy())


# In[16]:


# make a color map
from matplotlib.colors import LinearSegmentedColormap
colors = []
for l in np.linspace(1,0,100):
    colors.append((245/255,39/255,87/255,l))
for l in np.linspace(0,1,100):
    colors.append((24/255,196/255,93/255,l))
cm = LinearSegmentedColormap.from_list("shap", colors)


# In[17]:


len(shap_values)


# In[18]:


def fill_segmentation_original(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i] *10
    return out

import requests
r = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
feature_names = r.json()
def show_explanation(shap_values, img, inds):
#     img = Image.fromarray(img)

    segments_slic = slic(img, n_segments=49, compactness=1000, sigma=3)
    # plot our explanations
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(30,20)) 
    axes[0].imshow(img)
    axes[0].axis('off')
    max_val = np.max([np.max(np.abs(shap_values[i][:,:-1])) for i in range(len(shap_values))])
    for i in range(3):
        m = fill_segmentation_original(shap_values[inds[i]][0], segments_slic)
        axes[i+1].set_title(feature_names[str(inds[i])][1])
        axes[i+1].imshow(img, alpha=0.5)
        im = axes[i+1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
        axes[i+1].axis('off')
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
    cb.outline.set_visible(False)
    plt.show()
show_explanation(shap_values, img, top_preds[0])


# In[19]:


list(test_image_gen.class_indices.keys())


# In[20]:


# def extract_top_ten(shap_values, num_features=10):
#     new_shap_values=[]
#     for values in shap_values:
#         shap_list = list(set([item for sublist in values for item in sublist]))
#         shap_list.sort(reverse=True)
#         shap_list = shap_list[:num_features]
#         new_values = [numpy.asarray([a if a in shap_list else 0 for a in l]) for l in values] # filter out negative values and keep top 10
#         new_shap_values.append(new_values)
#     shap_values = numpy.asarray(new_shap_values)
#     return shap_values

# def predict_fn(x):
#     preds = model.predict(x)
#     p0 = 1 - preds
#     return np.hstack((p0, preds))

# def show_cut_image(shap_values, img_orig, num_features=10):
# # segment the image so we don't have to explain every pixel
#     segments_slic = slic(img_orig, n_segments=49, compactness=30, sigma=3)

#     # define a function that depends on a binary mask representing if an image region is hidden
#     def mask_image(zs, segmentation, image, background=None):
#         if background is None:
#             background = image.mean((0,1))
#         out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
#         for i in range(zs.shape[0]):
#             out[i,:,:,:] = image
#             for j in range(zs.shape[1]):
#                 if zs[i,j] == 0:
#                     out[i][segmentation == j,:] = background
#         return out
# #     def f(z):
# #         for img in mask_image(z, segments_slic, img_orig, 250):
# #             plt.imshow(img)
# #             plt.show()
# #         print(model.predict(mask_image(z, segments_slic, img_orig, 250)))
# #         return model.predict((mask_image(z, segments_slic, img_orig, 250)))

#     # # use Kernel SHAP to explain the network's predictions
# #     explainer = shap.KernelExplainer(f, np.zeros((1,50)))
# #     shap_values = explainer.shap_values(np.ones((1,50)), nsamples=1000) # runs model 1000 times

#     shap_values = extract_top_ten(shap_values, num_features)

#     # get the top predictions from the model
#     preds = predict_fn((np.expand_dims(img_orig.copy(), axis=0)))
#     top_preds = np.argsort(-preds)

#     # make a color map
#     from matplotlib.colors import LinearSegmentedColormap
#     colors = []
#     for l in np.linspace(1,0,100):
#         colors.append((0.2,0.2,0.2,l))
#     for l in np.linspace(0,1,100):
#         colors.append((0.5,0.5,0.5,l))
#     cm = LinearSegmentedColormap.from_list("shap", colors)

#     def fill_segmentation(values, segmentation):
#         out = np.zeros(segmentation.shape)
#         for i in range(len(values)):
#             out[segmentation == i] = 0 if values[i]> 0 else values[i]+0.5
#         return out

#     # plot our explanations
#     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,20))
#     inds = top_preds[0]
#     axes[0].imshow(img_orig)
#     axes[0].axis('off')
#     max_val = np.max([np.max(np.abs(shap_values[i][:,:-1])) for i in range(len(shap_values))])
#     for i in range(2):
#         m = fill_segmentation(shap_values[inds[i]][0], segments_slic)
#         axes[i+1].set_title(str(list(test_image_gen.class_indices.keys())[inds[i]]))
#         axes[i+1].imshow(img_orig)
#         im = axes[i+1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
#         plt.savefig('foo.png')
#         axes[i+1].axis('off')
#     cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
#     cb.outline.set_visible(False)
#     plt.show()
    
# show_cut_image(shap_values, img_orig)


# In[21]:


np.argsort(-preds)


# In[22]:


shap_values


# In[ ]:





# In[23]:


# make a color map
reload(grad)

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import cv2


# colors = []
# for l in np.linspace(1,0,100):
#     colors.append((0,0,0,l))
# for l in np.linspace(0,1,100):
#     colors.append((0,0,0,l))
# cm = LinearSegmentedColormap.from_list("shap", colors)

colors = []
for l in np.linspace(1,0,100):
    colors.append((245/255,39/255,87/255,l))
for l in np.linspace(0,1,100):
    colors.append((24/255,196/255,93/255,l))
cm = LinearSegmentedColormap.from_list("shap", colors)

def convert_to_shap_values(heatmap, verbose = 0):
    plt.matshow(heatmap)
    plt.axis('off')
    plt.margins(0,0)
    # cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
    # cb.outline.set_visible(False)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("heatmap.png", bbox_inches = 'tight',pad_inches = 0)
    plt.close()
#     img = image.load_img("heatmap.png", target_size=(7, 7, 1))
    
    fname = 'heatmap.png'
    image = Image.open(fname).convert("L")
    image.thumbnail((7,7))
    arr = np.asarray(image)
    if verbose:
        plt.imshow(arr, vmin=0, vmax=255)
        plt.show()
#     print(arr)
    
    shap_values = []
    for i in range (1000):
            shap_values.append([[item for sublist in arr for item in sublist]])
    shap_values = np.array(shap_values)
    shap_values = shap_values/np.max(shap_values)
#     print(shap_values)
    return shap_values

# def extract_top_ten(shap_values, num_features=10):
#     new_shap_values=[]
#     for values in shap_values:
#         shap_list = list(set([item for sublist in values for item in sublist]))
#         shap_list.sort(reverse=True)
#         shap_list = shap_list[:num_features]
#         new_values = [numpy.asarray([1 if a in shap_list else 0 for a in l]) for l in values] # filter out negative values and keep top 10
#         new_shap_values.append(new_values)
#     shap_values = numpy.asarray(new_shap_values)
#     return shap_values



def extract_top_ten(shap_values, num_features=10):
    shap_values_new = shap_values.copy()

    for values in shap_values_new:
        indices_of_features_to_zero = values[0].argsort()[:-num_features]
        values[0][indices_of_features_to_zero] = 0

    return shap_values_new


def init_csv(explainer_name):
    model_name = "pets"
    newrows_filename =  "{}_{}_rows.csv".format(model_name, explainer_name, model_name)
#     newrows_filename =  "{}-values.csv".format(model_name)
    if os.path.exists(newrows_filename):
        os.remove(newrows_filename)

    with open(newrows_filename, 'a', newline='') as fd:
        writer = csv.writer(fd)
        header = ["original_class", "original_probability", "new_probability",
                  "confidence_diff","class_change", "new_class",
                  "explainer","strategy", "sigma", "time", "num_features"]
        writer.writerow(header)

def save_row(original_class, original_probability, new_probability, confidence_diff ,class_change, new_class, explainer,strategy, sigma, time, num_features):
    model_name = "pets"
    fileName =  "{}_{}_rows.csv".format(model_name, explainer, model_name)
#     fileName =  "{}-values.csv".format(model_name)
    with open(fileName, 'a', newline='') as fd:
        writer = csv.writer(fd)
        result = [original_class, original_probability, new_probability, confidence_diff,class_change, new_class, explainer,strategy, sigma, time, num_features]
        writer.writerow(result)
        
# def fill_segmentation(values, segmentation, keep_top):
#         out = np.zeros((segmentation.shape))
#         for i in range(len(values)):
#             if keep_top:
#                 out[segmentation == i] = 0 if values[i]> 0 else 1
#             else:
#                 out[segmentation == i] = 1 if values[i]> 0 else 0
#         return out

    # define a function that depends on a binary mask representing if an image region is hidden
def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0,1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i,:,:,:] = image
        for j in range(zs.shape[1]):
            if zs[i,j] == 0:
                out[i][segmentation == j,:] = background
    return out

def mask_image_with_noise(zs, segmentation, image, sigma):
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    mask= np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i,:,:,:] = image
        mask[i,:,:,:] = random_noise(image/255, mode='gaussian', seed=42, var=sigma**2) *255 # tf.keras.preprocessing.image.img_to_array(random_noise(image))
        plt.imshow(mask[i,:,:,:])
        for j in range(zs.shape[1]):
            if zs[i,j] == 0:
                out[i][segmentation == j,:] = mask[i][segmentation == j,:]
    return out

def f_lime(z):
    return model.predict(preprocess_input(z.copy()))

prev_shap_values=None
prev_img_orig =None
prev_explanation = None
def calculate_predictions(img, img_orig, original_class_position, explainer_name, num_features, strategy, sigma, verbose = 0):
    # segment the image so we don't have to explain every pixel
    segments_slic = slic(img, n_segments=49, compactness=1000, sigma=3)
   
    def f(z):
        return model.predict(preprocess_input(mask_image(z, segments_slic, img_orig.copy(), 255)))
    
    new_class_lime = None
    new_prediction_lime = None
    global prev_shap_values
    global prev_img_orig
    global prev_explanation
    if ((explainer_name =='shap' or explainer_name =='random' or explainer_name =='grad') & (img_orig == prev_img_orig).all()):
        print ("Hitting cache, returning prev values")
        shap_values = prev_shap_values
    elif explainer_name =='shap':
        # use Kernel SHAP to explain the network's predictions
        explainer = shap.KernelExplainer(f, np.zeros((1,49)), start_label=1)
        shap_values = explainer.shap_values(np.ones((1,49)), nsamples=1000, start_label=1) # runs model 1000 times
    elif explainer_name =='grad':
        heatmap_orig = grad.make_gradcam_heatmap(img_orig.copy().reshape(1,224, 224, 3), model)
        heatmap =  cv2.resize(heatmap_orig, (7, 7))

        if verbose:
            plt.imshow(heatmap_orig)
            plt.show()
            plt.imshow(heatmap)
            plt.show()
            grad.test_drive_grad_original(img_orig, model)
        shap_values = []
        for i in range(1000):
            shap_values.append([[item for sublist in heatmap for item in sublist]])
        shap_values = np.array(shap_values)
        shap_values = shap_values / np.max(shap_values)
        
    elif explainer_name =='random':
        shap_values =[numpy.asarray([[random.uniform(0,1)  for iter in range(50)]]) for i in range(1000)]
    elif explainer_name == 'lime':
        if ((img_orig == prev_img_orig).all()):
            explanation = prev_explanation
            print ("Hitting LIME cache, returning prev values")
        else :
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(img_orig.astype("double"), f_lime, num_samples = 1000)
            prev_explanation = explanation
            prev_img_orig =img_orig
        
        print (explanation.top_labels)
        print(original_class_position)
#         if strategy == "top":
#             lime_img, _ = explanation.get_image_and_mask(explanation.top_labels[0] , positive_only=True, negative_only=False, hide_rest=True, num_features = num_features, min_weight=0)
#         else:
#             lime_img, _ = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, negative_only=True, num_features=1000, hide_rest=True)
#         new_class_lime = model.predict_classes(lime_img.reshape(1,224,224,3))[0]
#         new_prediction_lime = model.predict(preprocess_input(mask_image(z, segments_slic, img_orig.copy(), 255)))
        
        lime_img, mask = explanation.get_image_and_mask(explanation.top_labels[0] , positive_only=True, negative_only=False, hide_rest=True, num_features = num_features, min_weight=0)
        heatmap = cv2.resize(mask/255, (7, 7))
        
        if verbose :
            plt.matshow(lime_img/255)
            plt.show()
            plt.matshow(mask)
            plt.show()
            plt.matshow(heatmap)
            plt.show()
        
        shap_values = []
        for i in range (1000):
            shap_values.append([[item for sublist in heatmap for item in sublist]])
        shap_values = np.array(shap_values)
        shap_values = shap_values

    
    prev_shap_values=shap_values.copy()
    prev_img_orig =img_orig.copy()
    # get the top predictions from the model
    preds = model.predict(preprocess_input(np.expand_dims(img_orig.copy(), axis=0)))
    top_preds = np.argsort(-preds)
    inds = top_preds[0]

#     if verbose:
#         show_explanation(shap_values, img, inds)
    
    shap_values = [np.where(a<0, 0, a) for a in shap_values]
    shap_values = extract_top_ten(shap_values, num_features)
    
    if strategy == 'rest':
        shap_values = (shap_values - 1) * -1
    
    
    
    masked_image = mask_image_with_noise(shap_values[inds[0]], segments_slic, img_orig.copy(), sigma)[0]
    
    if verbose:
        plt.imshow((masked_image).astype(np.uint8))
        plt.show()
    
    prediction = model.predict(preprocess_input(np.expand_dims(masked_image.copy(), axis=0)))
    new_class  = decode_predictions(prediction)[0][0][1]
    new_prediction = prediction[0][original_class_position]
    
    if verbose:
        for pred in decode_predictions(prediction, 1000)[0]:
            if (pred[2] == new_prediction):
                print (pred)
    
    return new_class, new_prediction, new_class_lime, new_prediction_lime
    

def start(explainer_name, num_features_list, verbose = 0):
    counter = 0
#     for i in range(len(test_image_gen)):
#         batch = test_image_gen[i]
#         for j in range(len(batch[0])):
    for i in os.listdir(test_path + '/cats/')[:10]:
        
        counter = counter + 1
#         img = batch[0][j]
        image_path= test_path + '/cats/' +i
        img = image.load_img(image_path, target_size=(224, 224, 3))
        img_orig = image.img_to_array(img)
        pred = model.predict(preprocess_input(np.expand_dims(img_orig.copy(), axis=0)))
        original_class_position = np.argmax(pred[0])
        prediction = decode_predictions(pred)[0][0]
        original_class = prediction[1]
        original_confidence = prediction[2]
        print(original_class)
        if True :# original_class == batch[1][j]:
            if verbose:
                plt.imshow((img))
                plt.show()
                plt.close()
            for num_features in num_features_list:
                for strategy in ["top"]:
                    for sigma in [1]:
                        start_time = time.time()
                        new_class, new_prediction, _, _ = calculate_predictions(img, img_orig, original_class_position, explainer_name, num_features, strategy, sigma, verbose)
                        total_time = time.time() - start_time
                        print ("Explainer name: {} Image number: {} num_features {} strategy {}".format(explainer_name, counter, num_features, strategy))
                        print ("Original Class: {} original confidence:{} new class top:{} new confidence {}: {} "
                               .format(original_class, original_confidence, new_class, strategy, new_prediction) )

                        save_row(original_class, original_confidence, new_prediction,original_confidence - new_prediction ,original_class != new_class, new_class, explainer_name, strategy, sigma, total_time, num_features)

verbose= 1
for explainer_name in ["lime"]:
# for explainer_name in [sys.argv[1]]:
    init_csv(explainer_name)
    start(explainer_name, list(range(1,21)), verbose)


# In[ ]:


decode_predictions(model.predict(preprocess_input(np.expand_dims(img_orig.copy(), axis=0))))[0][0]


# In[ ]:


explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(img_orig.astype("double"), f_lime, num_samples = 1000)


# In[ ]:


lime_img, mask = explanation.get_image_and_mask(explanation.top_labels[0] , positive_only=True, negative_only=False, hide_rest=True, num_features = 10, min_weight=0)
shap_values = convert_to_shap_values(mask, verbose)

if verbose :
    plt.matshow(lime_img/255)
    plt.show()
    plt.matshow(mask)
    plt.show()
mask    


# In[ ]:


heatmap = cv2.resize(mask/255, (7, 7))
plt.imshow(heatmap)
plt.show()


# In[ ]:


help(cv2.resize)

