#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import tensorboard as tensorboard
import seaborn as seaborn
from tensorflow.python.client import device_lib
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
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


# In[ ]:





# In[2]:


data_dir="../../input/kaggle-flowers"
test_path= os.path.join(data_dir, 'test')

os.listdir(test_path)


# In[3]:


image_shape =(250, 250, 3)
test_gen = ImageDataGenerator(rescale =1./255)
batch_size=32
test_image_gen= test_gen.flow_from_directory(test_path, target_size=image_shape[:2], 
                                               color_mode='rgb', batch_size=batch_size, 
                                               class_mode='categorical',  shuffle=False)
test_image_gen.class_indices


# In[4]:


import shap
import numpy as np
X_test, _ = test_image_gen.next()
# background = X_test[np.random.choice(20, 10, replace=False)]
background = X_test[0:5]
# explain predictions of the model on three images
# e = shap.DeepExplainer(tf.keras.models.load_model('flowers'), background)

model =tf.keras.models.load_model('flowers-resnet')
# tulip_image_path = test_path + '/tulip/' + os.listdir(test_path + '/tulip/')[5]
tulip_image_path = test_path + '/tulip/' + os.listdir(test_path + '/tulip/')[5]
# plt.imshow(imread(dog_image))
img = image.load_img(tulip_image_path, target_size=(250, 250, 3))
# plt.imshow(imread(tulip_image_path))
img_orig = image.img_to_array(img)
standardized_image = test_gen.standardize(img_orig)
plt.imshow((standardized_image))


# In[5]:


os.listdir(test_path + '/sunflower/')


# In[ ]:





# In[ ]:





# In[6]:


# segment the image so we don't have to explain every pixel
segments_slic = slic(img, n_segments=49, compactness=30, sigma=3)

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
    return model.predict(mask_image(z, segments_slic, img_orig, 250))


# In[7]:


# use Kernel SHAP to explain the network's predictions
explainer = shap.KernelExplainer(f, np.zeros((1,50)))
shap_values = explainer.shap_values(np.ones((1,50)) , nsamples=1000) # runs VGG16 1000 times


# In[8]:


list(test_image_gen.class_indices.keys())


# In[9]:


# get the top predictions from the model
preds = model.predict(np.expand_dims(img_orig.copy(), axis=0))
top_preds = np.argsort(-preds)


# In[10]:


model.predict(np.expand_dims(img_orig.copy(), axis=0))


# In[11]:


plt.imshow(img_orig.copy())


# In[12]:


# make a color map
from matplotlib.colors import LinearSegmentedColormap
colors = []
for l in np.linspace(1,0,100):
    colors.append((245/255,39/255,87/255,l))
for l in np.linspace(0,1,100):
    colors.append((24/255,196/255,93/255,l))
cm = LinearSegmentedColormap.from_list("shap", colors)


# In[13]:


def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out

# plot our explanations
fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(30,20))
inds = top_preds[0]
axes[0].imshow(img)
axes[0].axis('off')
max_val = np.max([np.max(np.abs(shap_values[i][:,:-1])) for i in range(len(shap_values))])
for i in range(5):
    m = fill_segmentation(shap_values[inds[i]][0], segments_slic)
    axes[i+1].set_title(str(inds[i]))
    axes[i+1].imshow(img.convert('LA'), alpha=0.15)
    im = axes[i+1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
    axes[i+1].axis('off')
cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
cb.outline.set_visible(False)
plt.show()


# In[14]:


list(test_image_gen.class_indices.keys())


# In[21]:


def extract_top_ten(shap_values, num_features=10):
    new_shap_values=[]
    for values in shap_values:
        shap_list = list(set([item for sublist in values for item in sublist]))
        shap_list.sort(reverse=True)
        shap_list = shap_list[:num_features]
        new_values = [numpy.asarray([a if a in shap_list else 0 for a in l]) for l in values] # filter out negative values and keep top 10
        new_shap_values.append(new_values)
    shap_values = numpy.asarray(new_shap_values)
    return shap_values
    
def show_cut_image(shap_values, img_orig, num_features=1000):
# segment the image so we don't have to explain every pixel
    segments_slic = slic(img_orig, n_segments=49, compactness=1000, sigma=3)

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
    def f(z):
#         for img in mask_image(z, segments_slic, img_orig, 250):
#             plt.imshow(img)
#             plt.show()
#         print(model.predict(mask_image(z, segments_slic, img_orig, 250)))
        return model.predict((mask_image(z, segments_slic, img_orig, 250)))

    # # use Kernel SHAP to explain the network's predictions
    explainer = shap.KernelExplainer(f, np.zeros((1,50)))
    shap_values = explainer.shap_values(np.ones((1,50)), nsamples=5000) # runs model 1000 times

    shap_values = extract_top_ten(shap_values, num_features)

    # get the top predictions from the model
    preds = model.predict((np.expand_dims(img_orig.copy(), axis=0)))
    top_preds = np.argsort(-preds)

    # make a color map
    from matplotlib.colors import LinearSegmentedColormap
    colors = []
    for l in np.linspace(1,0,100):
        colors.append((0.2,0.2,0.2,l))
    for l in np.linspace(0,1,100):
        colors.append((0.5,0.5,0.5,l))
    cm = LinearSegmentedColormap.from_list("shap", colors)

    def fill_segmentation(values, segmentation):
        out = np.zeros(segmentation.shape)
        for i in range(len(values)):
            out[segmentation == i] = 0 if values[i]> 0 else 1
        return out

    # plot our explanations
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(30,20))
    inds = top_preds[0]
    axes[0].imshow(img_orig)
    axes[0].axis('off')
    max_val = np.max([np.max(np.abs(shap_values[i][:,:-1])) for i in range(len(shap_values))])
    for i in range(5):
        m = fill_segmentation(shap_values[inds[i]][0], segments_slic)
        axes[i+1].set_title(str(list(test_image_gen.class_indices.keys())[inds[i]]))
        axes[i+1].imshow(img_orig)
        im = axes[i+1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
        plt.savefig('foo.png')
        axes[i+1].axis('off')
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
    cb.outline.set_visible(False)
    plt.show()
    
show_cut_image(shap_values, img_orig)


# In[17]:


np.argsort(-preds)


# In[ ]:





# In[19]:


import grad_cam as grad
from importlib import reload


# In[27]:


# make a color map
reload(grad)

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image



colors = []
for l in np.linspace(1,0,100):
    colors.append((0,0,0,l))
for l in np.linspace(0,1,100):
    colors.append((0,0,0,l))
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
    for i in range (5):
            shap_values.append([[item for sublist in arr for item in sublist]])
    shap_values = np.array(shap_values)
    shap_values = shap_values/np.max(shap_values)
#     print(shap_values)
    return shap_values

def extract_top_ten(shap_values, num_features=10):
    new_shap_values=[]
    for values in shap_values:
        shap_list = list(set([item for sublist in values for item in sublist]))
        shap_list.sort(reverse=True)
        shap_list = shap_list[:num_features]
        new_values = [numpy.asarray([1 if a in shap_list else 0 for a in l]) for l in values] # filter out negative values and keep top 10
        new_shap_values.append(new_values)
    shap_values = numpy.asarray(new_shap_values)
    return shap_values

def init_csv(explainer_name):
    model_name = "flowers"
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
    model_name = "flowers"
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
        mask[i,:,:,:] = random_noise(image, mode='s&p', seed=42, amount=1)#, var=sigma**2) # tf.keras.preprocessing.image.img_to_array(random_noise(image))
        for j in range(zs.shape[1]):
            if zs[i,j] == 0:
                out[i][segmentation == j,:] = mask[i][segmentation == j,:]
    return out

prev_shap_values=None
prev_img_orig =None
prev_explanation = None
def calculate_predictions(img_orig, original_class, explainer_name, num_features, strategy, sigma, verbose = 0):
    # segment the image so we don't have to explain every pixel
    segments_slic = slic(img_orig, n_segments=49, compactness=1000, sigma=3)
   
    def f(z):
        return model.predict(mask_image(z, segments_slic, img_orig, 255))
    
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
        explainer = shap.KernelExplainer(f, np.zeros((1,49)))
        shap_values = explainer.shap_values(np.ones((1,49)), nsamples=1000) # runs model 1000 times
    elif explainer_name =='grad':
        # use Kernel SHAP to explain the network's predictions
        last_conv_layer_name = "conv2d"
        classifier_layer_names = [            "global_average_pooling2d",            "dense_1",        ]
#         last_conv_layer_name = "conv2d_2"
#         classifier_layer_names = [            "global_average_pooling2d_2",            "dense_5",        ]
        
        heatmap = grad.make_gradcam_heatmap(img_orig.reshape(1,250, 250, 3), model, last_conv_layer_name, classifier_layer_names)
        
        if verbose:
            grad.test_drive_grad_original(img_orig)
            plt.imshow(heatmap)
            plt.show()
            
        shap_values = []
        for i in range(5):
            shap_values.append([[item for sublist in heatmap for item in sublist]])
        shap_values = np.array(shap_values)
        shap_values = shap_values / np.max(shap_values)
        
    elif explainer_name =='random':
        shap_values =[numpy.asarray([[random.uniform(0,1)  for iter in range(50)]]) for i in range(5)]
    elif explainer_name == 'lime':
        if ((img_orig == prev_img_orig).all()):
            explanation = prev_explanation
            print ("Hitting LIME cache, returning prev values")
        else :
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(img_orig.astype("double"), model.predict, num_samples = 1000)
            prev_explanation = explanation
            prev_img_orig =img_orig
        
        if strategy == "top":
            lime_img, _ = explanation.get_image_and_mask(explanation.top_labels[original_class] , positive_only=True, negative_only=False, hide_rest=True, num_features = num_features, min_weight=0)
        else:
            lime_img, _ = explanation.get_image_and_mask(explanation.top_labels[original_class], positive_only=False, negative_only=True, num_features=1000, hide_rest=True)
        
        if verbose :
            plt.matshow(lime_img)
            plt.show()
        
        new_class_lime = model.predict_classes(lime_img.reshape(1,250,250,3))[0]
        new_prediction_lime = model.predict(lime_img.reshape(1,250,250,3))[0][original_class]
        
        _, mask = explanation.get_image_and_mask(explanation.top_labels[original_class] , positive_only=True, negative_only=False, hide_rest=True, num_features = num_features, min_weight=0)
        shap_values = convert_to_shap_values(mask, verbose)
        
    prev_shap_values=shap_values.copy()
    prev_img_orig =img_orig.copy()
    # get the top predictions from the model
    preds = model.predict(np.expand_dims(img_orig.copy(), axis=0))
    top_preds = np.argsort(-preds)
    inds = top_preds[0]

#     show_cut_image(shap_values, img_orig, 20)
    
    shap_values = [np.where(a<0, 0, a) for a in shap_values]
    shap_values = extract_top_ten(shap_values, num_features)
    
    if strategy == 'rest':
        shap_values = (shap_values - 1) * -1
    
    masked_image = mask_image_with_noise(shap_values[inds[0]], segments_slic, img_orig, sigma)
    
    if verbose:
        plt.imshow(masked_image[0])
        plt.show()
    
    new_class  = model.predict_classes(masked_image)[0]
    new_prediction = model.predict(masked_image)[0][original_class]
    return new_class, new_prediction, new_class_lime, new_prediction_lime
    

def start(explainer_name, num_features_list, verbose = 0):
    counter = 0
    for i in range(len(test_image_gen)):
        batch = test_image_gen[i]
        for j in range(len(batch[0])):
            counter = counter + 1
            img = batch[0][j]
            original_class = model.predict_classes(img.reshape(1,250,250,3))[0]
            original_confidence = model.predict(img.reshape(1,250,250,3))[0][original_class]
            if original_class == np.argmax(batch[1][j]):
                if verbose:
                    plt.imshow((img))
                    plt.show()
                    plt.close()
                for num_features in num_features_list:
                    for strategy in ["top", "rest"]:
                        for sigma in [1]:
                            if explainer_name == 'lime':
                                print ("Explainer name: {} Image number: {} num_features {} batch {} number {} strategy {}".format(explainer_name, counter, num_features, i, j, strategy))
                                start_time = time.time()
                                new_class, new_prediction, new_class_lime, new_prediction_lime = calculate_predictions(img, original_class, explainer_name, num_features, strategy, sigma, verbose)
                                total_time = time.time() - start_time
                                print ("Lime Standard Original Class: {} original confidence:{} new class top:{} new confidence {}: {} "
                                       .format(original_class, original_confidence, new_class, strategy, new_prediction) )

                                print ("Lime Original: Original Class: {} original confidence:{} new class top:{} new confidence {}: {} "
                                       .format(original_class, original_confidence, new_class_lime, strategy, new_prediction_lime) )

                                save_row(original_class, original_confidence, new_prediction,original_confidence - new_prediction ,original_class != new_class, new_class, "lime-standard", strategy, sigma, total_time, num_features)
                                save_row(original_class, original_confidence, new_prediction_lime,original_confidence - new_prediction_lime ,original_class != new_class_lime, new_class_lime, "lime", strategy, sigma, total_time, num_features)
                            else:
                                print ("Explainer name: {} Image number: {} num_features {} batch {} number {} strategy {}".format(explainer_name, counter, num_features, i, j, strategy))
                                start_time = time.time()
                                new_class, new_prediction, _, _ = calculate_predictions(img, original_class, explainer_name, num_features, strategy, sigma, verbose)
                                total_time = time.time() - start_time
                                print ("Original Class: {} original confidence:{} new class top:{} new confidence {}: {} "
                                       .format(original_class, original_confidence, new_class, strategy, new_prediction) )

                                save_row(original_class, original_confidence, new_prediction,original_confidence - new_prediction ,original_class != new_class, new_class, explainer_name, strategy, sigma, total_time, num_features)

verbose= 0
# for explainer_name in ["grad"]:
for explainer_name in [sys.argv[1]]:
    if explainer_name == "lime":
        init_csv("lime-standard")
        
    init_csv(explainer_name)
    start(explainer_name, list(range(1,21)), verbose)


# In[26]:


model.summary()


# In[ ]:




