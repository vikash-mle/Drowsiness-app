#!/usr/bin/env python
# coding: utf-8

# ## TITLE:   DROWSINESS DETECTION model 

# ##### BY  Beta_Hacker:- Vikash Ranjan, Bibhutibhusan Nayak, Sahin ahmed

# # INTRODUCTION :

# ### A lot of  drivers feel lazy or sleepy some times which could lead to fatal accidents.  Various studies states that around 30-40% accidents occur due to drowsy driver.For increse the work efficiency in job and to   reduce these road accidents, a system should be developed which can identify the expressions of the driver by the car camera setup and then alert the person in advance. This could save a lot of lives. ​This project is focused on  detecting drowsiness of the of a person based on images and alert them before hand. Therefore,  a AI  model is built which can help achieve this with sufficient accuracy and safety.​

# # PROBLEM STATEMENT:

# #### TO PREDICT THE PHYSICAL CONDITION OF THE DRIVER BASED ON TRAINING IMAGES GIVEN WITH OPEN EYE,CLOSED EYE, YAWNING,NO YAWNING, WITH BETTER ACCURACY AND LESS TIME.

# ## METHODOLOGY:

# ### Our solution to this problem is to build a detection system that identifies key attributes of drowsiness and     triggers an alert   when someone is drowsy before it is too late.
# ### We followed the below methodlogy for our solution 
# ### 1.VISUALISE AND INSPECT THE DATA
# ### 2.PREPROCESS THE DATA AND PREPARE FOR CNN MODEL.
# ### 3.TRAIN DIFFERENT MODELS AND EVALUATE THEIR PERFORMANCE.
# ### 4.ANALYSE THE RESULTS OBTAINED.
# ### 5.FINALLY DRAW IMPORTANT CONCLUSION .

# # DATASET USED:

# ### LINK: https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new
# ### IMAGE CLASSES/LABELS:['Closed', 'no_yawn', 'Open', 'yawn']
# ### EACH CLASS HAS 617 IMAGES IN TRAINING SET
# ### EACH CLASS HAS 109 IMAGES IN TEST SET

# ### HERE EYES OPEN , NO YWAN MEANS PERSON IS NOT DROWSY.
# ### HERE EYES CLOSED, YAWN MEANS PERSON IS DRWOSY 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2


# In[2]:


get_ipython().system('pip install pandas')


# # VISUALISING THE DATA

# In[3]:


#GETTING IMAGE LABELS
labels = os.listdir(r"C:\Users\Sai_Pg_Lab_WS\Downloads\dataset_new\train")


# In[4]:


labels


# In[5]:


import os

# Walk through pizza_steak directory and list number of files
for dirpath, dirnames, filenames in os.walk("dataset"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# In[6]:


import pathlib
import numpy as np
data_dir = pathlib.Path(r"C:\Users\Sai_Pg_Lab_WS\Downloads\dataset_new\train") # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
print(class_names)


# ## FUNCTION TO VIEW A RANDOM IMAGE

# In[7]:


# View an image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def view_random_image(target_dir, target_class):
  # Setup target directory (we'll view images from here)
  target_folder = target_dir+target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  print(f"Image shape: {img.shape}") # show the shape of the image

  return img


# In[8]:


img = view_random_image(target_dir=r"C:\Users\Sai_Pg_Lab_WS\Downloads\dataset_new\train/",
                        target_class="Open")


# In[9]:


img = view_random_image(target_dir=r"C:\Users\Sai_Pg_Lab_WS\Downloads\dataset_new\train/",
                        target_class="Closed")


# In[10]:


img = view_random_image(target_dir=r"C:\Users\Sai_Pg_Lab_WS\Downloads\dataset_new\train/",
                        target_class="Yawn")


# In[11]:


img = view_random_image(target_dir=r"C:\Users\Sai_Pg_Lab_WS\Downloads\dataset_new\train/",
                        target_class="No_yawn")


# In[ ]:





# In[12]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img(r"C:\Users\Sai_Pg_Lab_WS\Downloads\dataset_new\train\yawn\78.jpg")  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=r'C:\Users\Sai_Pg_Lab_WS\Downloads\dataset_new\New folder', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely


# ### We can see that images are of different shape, they are colored and of different people driving with eyes open,closed,yawning and no yawning

# # FUNCTION TO CREATE TRAINING DATA WITH LABELS

# In[13]:


Datadirectory = r"C:\Users\Sai_Pg_Lab_WS\Downloads\dataset_new\train/"
#training_data = []
img_size = 224
training_data = []
def create_training_data(): 
    for category in labels:
      path = os.path.join(Datadirectory, category)
      class_num = labels.index(category)
      for img in os.listdir(path):
           img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
           backtorgb = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
           new_array = cv2.resize(backtorgb, (img_size,img_size))
           training_data.append([new_array, class_num])
       


# In[14]:


create_training_data()


# In[15]:


training_data


# In[16]:


len(training_data)


# In[17]:


import random
random.shuffle(training_data)


# In[18]:


#here we reshape the image.
X = []
y = []
for features, label in training_data:
  X.append(features)
  y.append(label)


# In[19]:


X=np.array(X)
y=np.array(y)


# In[20]:


from matplotlib import pyplot as plt
plt.imshow((x[0][0])/255, interpolation='nearest')
plt.show()


# In[21]:


X.shape


# In[22]:


y.shape


# In[23]:


X = np.array(X).reshape(-1, img_size, img_size, 3)


# In[24]:


X.shape


# In[25]:


y


# In[26]:


#RESCALE THE TRAIN DATA
X=X/255


# # FUNCTION TO CREATE TEST DATA

# In[27]:


Datadirectory = r"C:\Users\Sai_Pg_Lab_WS\Downloads\dataset_new\test/"
test_data = []
img_size = 224

def create_test_data():
    for category in labels:
      path = os.path.join(Datadirectory, category)
      class_num = labels.index(category)
      for img in os.listdir(path):
           img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
           backtorgb = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
           new_array = cv2.resize(backtorgb, (img_size,img_size))
           test_data.append([new_array, class_num])


# In[28]:


create_test_data()


# In[29]:


import random
random.shuffle(test_data)


# In[30]:


#here we reshape the image.
X_test = []
y_test = []
for features, label in test_data:
  X_test.append(features)
  y_test.append(label)


# In[31]:


X_test=np.array(X_test)
y_test=np.array(y_test)


# In[32]:


X_test=X_test/255


# In[33]:


X_test.shape


# # MODEL BUILDING:

# ## 1.BASELINE MODEL: we first built a simple model without data augmentation.

# In[34]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

# Create our model (a clone of model_8, except to be multi-class)
model_1 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(pool_size=(2, 2)),
  Conv2D(10, 3, activation='relu'),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(4, activation='softmax') # changed to have 10 neurons (same as number of classes) and 'softmax' activation
])

# Compile the model
model_1.compile(loss="sparse_categorical_crossentropy", # changed to categorical_crossentropy
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


# In[35]:


train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
        shear_range=0.2)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
        shear_range=0.2)
#train_generator.fit(X)
#test_generator.fit(X_test)
train_generator = train_generator.flow(X, y, shuffle=False)
test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)


# In[36]:


model_1.summary()


# In[37]:


# Fit the model
tf.debugging.set_log_device_placement(True)

# Place tensors on the CPU
with tf.device('/GPU:0'):
    history_1 = model_1.fit(train_generator,batch_size=1,
                            epochs=20,validation_data=test_generator)


# In[38]:


#FUNCTION TO PLOT THE HISTORY
def plot_history(history, name):
    with plt.xkcd(scale=0.2):
      fig, ax = plt.subplots(1,2, figsize=(12,6))
      for i, metric in enumerate(['loss', 'accuracy']): 
          ax[i].plot(history.history[metric], label='Train', color='#EFAEA4',linewidth=3)
          ax[i].plot(history.history[f'val_{metric}'], label='Validation', color='#B2D7D0',linewidth=3)
          if metric == 'accuracy': 
            ax[i].axhline(0.5, color='#8d021f', ls='--', label='Trivial accuracy')
            ax[i].set_ylabel("Accuracy", fontsize=14)
          else:
            ax[i].set_ylabel("Loss", fontsize=14)
          ax[i].set_xlabel('Epoch', fontsize=14)

      plt.suptitle(f'{name} Training', y=1.05, fontsize=16)
      plt.legend(loc='best')
      plt.tight_layout()
plot_history(history_1,'baseline model')


# ### we can see that the baseline model is performing fairly well with 96.19% train accuracy and 85.83 percent validation accuracy

# # Evaluation of base model

# In[39]:



y_pred=[]

for i in X_test:
    y_pred.append(model_1.predict(np.expand_dims(i,axis=0)).argmax())
    


# In[40]:


get_ipython().system('pip install seaborn')


# In[41]:


import seaborn as sns


# In[42]:


from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test,y_pred)


# In[43]:


sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='')


# In[44]:


print(classification_report(y_test,y_pred))


# In[ ]:





# # PLOTTING FEATURE MAPS:
# 

# *The feature maps of a CNN capture the result of applying the filters to an input image . I.e at each layer, the feature map is the output of that layer. The reason for visualising a feature map for a specific input image is to try to gain some understanding of what features our CNN detects.*

# In[45]:


#plotting feature map function
def plot_featuremaps(img,activations,layer_names):
    fig, axs = plt.subplots(ncols=4, nrows=3,figsize = (6,6))
    gs = axs[1, 2].get_gridspec()
    # remove the underlying axes
    for ax in axs[1:-1, 1:-1]:
        ax[0].remove()
        ax[1].remove()
    axbig = fig.add_subplot(gs[1:-1, 1:-1])

    axbig.imshow(img.squeeze() + 0.5)
#     axbig.set_title(f'{cifar10dict[np.argmax(model.predict(img))]}')
    axbig.axis('off')

    for i, axis in enumerate(axs.ravel()):
        axis.imshow(activations.squeeze()[:,:,i-2], cmap='gray')
        axis.axis('off')

    fig.tight_layout()
    fig.suptitle(f'Feature maps for {layer_names[0]}',size=16,y=1.05);


# In[46]:


#feature map for first convolutional layer
layer0_output = model_1.layers[0].output
activation_model = tf.keras.Model(inputs = model_1.input, outputs = layer0_output)
img = X_test[1].reshape(-1,224,224,3)
activations = activation_model.predict(img)
plot_featuremaps(img,activations,[model_1.layers[0].name])


# In[47]:


#feature map for second convolutional layer
layer0_output = model_1.layers[1].output
activation_model = tf.keras.Model(inputs = model_1.input, outputs = layer0_output)
img = X_test[1].reshape(-1,224,224,3)
activations = activation_model.predict(img)
plot_featuremaps(img,activations,[model_1.layers[1].name])


# In[48]:


#feature map for  maxpool layer
layer0_output = model_1.layers[2].output
activation_model = tf.keras.Model(inputs = model_1.input, outputs = layer0_output)
img = X_test[1].reshape(-1,224,224,3)
activations = activation_model.predict(img)
plot_featuremaps(img,activations,[model_1.layers[2].name])


# In[49]:


##feature map for 4th convolutional layer
layer0_output = model_1.layers[3].output
activation_model = tf.keras.Model(inputs = model_1.input, outputs = layer0_output)
img = X_test[1].reshape(-1,224,224,3)
activations = activation_model.predict(img)
plot_featuremaps(img,activations,[model_1.layers[3].name])


# In[50]:


#feature map for last convolutional layer
layer0_output = model_1.layers[4].output
activation_model = tf.keras.Model(inputs = model_1.input, outputs = layer0_output)
img = X_test[1].reshape(-1,224,224,3)
activations = activation_model.predict(img)
plot_featuremaps(img,activations,[model_1.layers[4].name])


# # INFERENCES FROM FEATURE MAPS:
# we can observe that each convoultion layer is detecting the important features of images very well which is why model is performing  well.

# # PLOTTING GRADCAM VISUALISATION:
# One way to  visualizing what CNNs are actually looking at, is by  using Grad-CAM. Gradient weighted Class Activation Map (Grad-CAM) produces a heat map that highlights the important regions of an image by using the gradients of the target of the final convolutional layer. 

# In[51]:


#source-https://keras.io/examples/vision/grad_cam/
# Function to expand the dimension of the input image
def get_img_array(img_array, size):
    array = np.expand_dims(img_array, axis=0)
    return array

#Function to plot GradCam
def make_gradcam_heatmap(img_array, model,conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# In[52]:


img_array =X_test[1]
conv_layer_name = 'conv2d_3'
# Make model

# Remove last layer's softmax
model_1.layers[-1].activation=None

# Print what the top predicted class is
preds = model_1.predict(np.expand_dims(X_test[1],axis=0))
print("Predicted:",preds)

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(np.expand_dims(img_array,axis=0)*255, model_1,conv_layer_name)
#saliency=saliency_graphs(model, np.expand_dims(x_test_data[0][0], axis=0))
# Display heatmap
fig,ax = plt.subplots(1,2, figsize=(14,8))
ax[0].imshow(X_test[1])
ax[1].matshow(heatmap)
ax[1].set_title('Heat-map from Grad-CAM')
ax[0].set_title('Original Image')
fig.suptitle('Original Image vs. GRAD CAM', fontsize = 24)
plt.show()


# FROM THE GRAD CAM VISUALISATION WE CAN CLEARLY UNDERSTAND WHAT OUR MODEL IS SEEING IN TRAINING IMAGES. 

# In[53]:


model_1.save('model1.h5')


# In[54]:


model_1.save_weights('model1weights.h5')


# # PLOTTING SALIENCY MAP:
# Saliency Map is an image in which the brightness of a pixel represents how salient the pixel is i.e brightness of a pixel is directly proportional to its saliency. It is generally a grayscale image. Saliency maps are also called as a heat map where hotness refers to those regions of the image which have a big impact on predicting the class which the object belongs to. 
# The purpose of the saliency map is to find the regions which are prominent or noticeable at every location in the visual field and to guide the selection of attended locations, based on the spatial distribution of saliency. 
# source:https://www.geeksforgeeks.org/what-is-saliency-map/

# In[55]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from matplotlib import cm
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.activation_maximization.callbacks import Progress
from tf_keras_vis.activation_maximization.input_modifiers import Jitter, Rotate2D
from tf_keras_vis.activation_maximization.regularizers import TotalVariation2D, Norm
from tf_keras_vis.utils.model_modifiers import ExtractIntermediateLayer, ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore


# In[56]:


#source:https://raghakot.github.io/keras-vis/visualizations/saliency/
def plot_smoothgrad_of_a_model(model, X):
    score = CategoricalScore([3])

    # Create Saliency visualization object
    saliency = Saliency(model,
                        model_modifier=ReplaceToLinear(),
                        clone=True)

    # Generate saliency map
    saliency_map = saliency(score, X.reshape(-1,224,224,3),
                            smooth_samples=20, # The number of calculating gradients iterations
                            smooth_noise=0.20) # noise spread level

    # Render
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,8))
    #for i, title in enumerate(image_titles):
        #ax[i].set_title(title, fontsize=16)
    ax[0].imshow(X)
    ax[0].set_title('original image')
    ax[1].imshow(saliency_map[0],cmap='jet')
    ax[1].set_title('saliency map')
    fig.suptitle('Original Image vs. Saliency MAP', fontsize = 24)
    #plt.tight_layout()
    plt.show()


# In[57]:


plot_smoothgrad_of_a_model(model_1,X_test[2])


# # SECOND MODEL:

# In[58]:


import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD


# In[59]:



model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=X.shape[1:]))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
model.add(Conv2D(128, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4,activation='softmax'))
model.summary()


# In[60]:


model.compile(loss="sparse_categorical_crossentropy", # changed to categorical_crossentropy
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


# In[61]:


# Fit the model
history_1 = model.fit(train_generator,
                        epochs=50,batch_size=1,validation_data=test_generator)


# In[62]:


model.save('model2.h5')


# In[63]:


plot_history(history_1, 'model_2')


# In[64]:


layer0_output = model.layers[0].output
activation_model = tf.keras.Model(inputs = model.input, outputs = layer0_output)
img = X_test[1].reshape(-1,224,224,3)
activations = activation_model.predict(img)
plot_featuremaps(img,activations,[model.layers[0].name])


# In[65]:


layer0_output = model.layers[1].output
activation_model = tf.keras.Model(inputs = model.input, outputs = layer0_output)
img = X_test[1].reshape(-1,224,224,3)
activations = activation_model.predict(img)
plot_featuremaps(img,activations,[model.layers[1].name])


# In[66]:


layer0_output = model.layers[4].output
activation_model = tf.keras.Model(inputs = model.input, outputs = layer0_output)
img = X_test[1].reshape(-1,224,224,3)
activations = activation_model.predict(img)
plot_featuremaps(img,activations,[model.layers[4].name])


# In[67]:


plot_smoothgrad_of_a_model(model,X_test[2])


# In[68]:


#model = tf.keras.applications.mobilenet.MobileNet()


# # TRANSFER LEARNING METHOD

# The reuse of a previously learned model on a new problem is known as transfer learning. It’s particularly popular in deep learning right now since it can train deep neural networks with a small amount of data.
# 
# 
# Transfer learning offers a number of advantages, the most important of which are reduced training time, improved neural network performance (in most circumstances), and the absence of a large amount of data.
# 
# 

# # INCEPTION RESNET V2 MODEL:
# Inception-ResNet-v2 is a convolutional neural architecture that builds on the Inception family of architectures but incorporates residual connections (replacing the filter concatenation stage of the Inception architecture).
# 
# 
# SOURCE: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/viewer.html?pdfurl=https%3A%2F%2Farxiv.org%2Fpdf%2F1512.00567v3.pdf&clen=517626&chunk=true

# In[69]:


base_model = tf.keras.applications.InceptionResNetV2(
                     include_top=False,
                     weights='imagenet',
                     input_shape=(299,299,3)
                     )
  
base_model.trainable=False
# For freezing the layer we make use of layer.trainable = False
# means that its internal state will not change during training.
# model's trainable weights will not be updated during fit(),
# and also its state updates will not run.
  
model_3 = tf.keras.Sequential([ 
        base_model,   
        tf.keras.layers.BatchNormalization(renorm=True),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])


# In[70]:


model_3.summary()


# In[71]:


model_3.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# categorical cross entropy is taken since its used as a loss function for
# multi-class classification problems where there are two or more output labels.
# using Adam optimizer for better performance
# other optimizers such as sgd can also be used depending upon the model


# In[73]:


history_3=model_3.fit(train_generator,epochs=20,batch_size=1,validation_data=test_generator, shuffle=True)


# In[74]:


plot_history(history_3, 'model_3')


# In[75]:


model_3.save('resnet2.h5')


# In[76]:


model_3.save_weights('resnetweights.h5')


# # IMAGE AUGMENTATION

#  Image augmentation is a technique of altering the existing data to create some more data for the model training process. 
# In other words, it is the process of artificially expanding the available dataset for training a deep learning model.
# 
# AUGMENTATION USED:
# We have used following data augmentation techniques on our dataset:      
# 1. brightness_range - To generate images for night time/low light condition so that the model is able to perform even in such conditions. 
# 2. Roation: Drowsiness or other traffic issues can lead to driver's head bending by few degrees, to be able to capture that in our model we have used this method.  
# 3. zca_whitening - It helps remove redundent pixels from images and make them more detailed making it easy for the model to learn features.   
# 4. Horizontal and Vertical shift - Varying heights of drivers, vehicle condition etc can lead to face not being in front of the camera making it difficult for it to detect drowsiness. 
# 5. Zooming - Drivers depending on their height may be sitting too close or far away from the camera hence it has to be taken into account for the model to be able to learn better.

# In[77]:


get_ipython().system('pip install SciPy')
import scipy  


# In[78]:


train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
#train_generator.fit(X)
#test_generator.fit(X_test)
train_generator = train_generator.flow(X, y, shuffle=False)
test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)


# In[ ]:





# In[79]:


base_model = tf.keras.applications.InceptionResNetV2(
                     include_top=False,
                     weights='imagenet',
                     input_shape=(224,224,3)
                     )
  
base_model.trainable=False
# For freezing the layer we make use of layer.trainable = False
# means that its internal state will not change during training.
# model's trainable weights will not be updated during fit(),
# and also its state updates will not run.
  
model_4 = tf.keras.Sequential([ 
        base_model,   
        tf.keras.layers.BatchNormalization(renorm=True),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])


# In[80]:


model_4.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# categorical cross entropy is taken since its used as a loss function for
# multi-class classification problems where there are two or more output labels.
# using Adam optimizer for better performance
# other optimizers such as sgd can also be used depending upon the model


# In[81]:


history_4 = model_4.fit(train_generator, epochs=10, validation_data=test_generator, shuffle=True)


# In[82]:


history_4=plot_history(history_4, 'model_4')


# In[83]:


model_4.save('resnet2_with_aug.h5')
model_4.save_weights('resnet2_with_aug_weights.h5')


# In[84]:


import cv2
import numpy as np


# In[85]:


from tensorflow.keras.models import load_model


# In[86]:


load=load_model(r"C:\Users\Sai_Pg_Lab_WS\Documents\New folder (2)\resnet2_with_aug.h5")


# In[87]:


img=cv2.resize(cv2.imread(r"C:\Users\Sai_Pg_Lab_WS\Downloads\dataset_new\test\Closed\_284.jpg"),(224,224))
img=np.reshape(img,[1,224,224,3])


# In[88]:


classes =load.predict(img)
classes
# names = [class_names[i] for i in classes]
# print(names)


# In[89]:


labels


# In[90]:


(X_test[0].shape)


# In[91]:


y_pred=[]

for i in X_test:
    y_pred.append(model_4.predict(np.expand_dims(i,axis=0)).argmax())


# In[92]:


confusion_matrix(y_test,y_pred)


# In[93]:


sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='')


# In[94]:


print(classification_report(y_test,y_pred))


# In[95]:


imageDataGenerator_obj = tf.keras.preprocessing.image.ImageDataGenerator(brightness_range=[0.1,2], zca_whitening= True,
                                            width_shift_range= 0.2, height_shift_range = 0.2, 
                                            rotation_range= 0.3, zoom_range = 0.3)
iterator = imageDataGenerator_obj.flow(X, batch_size=1)
plt.figure(figsize = (10,10))
for j in range(9):

    plt.subplot(330 + 1 + j)
    
    chunk = iterator.next()

    sub_img = chunk[0].astype('uint8')

    plt.imshow(sub_img) 
plt.show()


# In[96]:


imageDataGenerator_obj = tf.keras.preprocessing.image.ImageDataGenerator(brightness_range=[0.1,2],horizontal_flip=True,zoom_range=0.3)
iterator = imageDataGenerator_obj.flow(X, batch_size=32)
plt.figure(figsize = (10,10))
for j in range(9):

    plt.subplot(330 + 1 + j)
    
    chunk = iterator.next()

    sub_img = chunk[0].astype('uint8')

    plt.imshow(sub_img) 
plt.show()


# In[97]:


imageDataGenerator_obj = tf.keras.preprocessing.image.ImageDataGenerator(brightness_range=[0.1,2],horizontal_flip=True)
iterator = imageDataGenerator_obj.flow(X, batch_size=32)
plt.figure(figsize = (10,10))
for j in range(9):

    plt.subplot(330 + 1 + j)
    
    chunk = iterator.next()

    sub_img = chunk[0].astype('uint8')

    plt.imshow(sub_img) 
plt.show()


# In[98]:


import os
import cv2


# ### Model with taking only face images 

# In[99]:


def face_for_yawn(direc=r"C:\Users\Sai_Pg_Lab_WS\Downloads\dataset_new\train", face_cas_path=r"C:\Users\Sai_Pg_Lab_WS\Downloads\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml"):
    yaw_no = []
    IMG_SIZE = 145
    categories = ["yawn", "no_yawn"]
    for category in categories:
        path_link = os.path.join(direc, category)
        class_num1 = categories.index(category)
        print(class_num1)
        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
            face_cascade = cv2.CascadeClassifier(face_cas_path)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_color = img[y:y+h, x:x+w]
                resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
                yaw_no.append([resized_array, class_num1])
    return yaw_no


yawn_no_yawn = face_for_yawn()


# ### Reference for cascade- https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

# In[100]:


def get_data(dir_path=r"C:\Users\Sai_Pg_Lab_WS\Downloads\dataset_new\train", face_cas=r"C:\Users\Sai_Pg_Lab_WS\Downloads\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml", eye_cas=r"C:\Users\Sai_Pg_Lab_WS\Downloads\opencv-master\data\haarcascades\haarcascade_eye.xml"):
    labels = ['Closed', 'Open']
    IMG_SIZE = 145
    data = []
    for label in labels:
        path = os.path.join(dir_path, label)
        class_num = labels.index(label)
        class_num +=2
        print(class_num)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                print(e)
    return data


# In[101]:


import numpy as np


# In[102]:


data_train = get_data()
def append_data():
#     total_data = []
    yaw_no = face_for_yawn()
    data = get_data()
    yaw_no.extend(data)
    return np.array(yaw_no)
new_data = append_data()


# In[103]:


X = []
y = []
for features, label in new_data:
  X.append(features)
  y.append(label)


# In[104]:


X = np.array(X)
X = X.reshape(-1, 145, 145, 3)


# In[105]:


from sklearn.preprocessing import LabelBinarizer
label_bin = LabelBinarizer()
y = label_bin.fit_transform(y)
y = np.array(y)


# In[106]:


from sklearn.model_selection import train_test_split
seed = 42
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)


# In[107]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense,Dropout
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD


# In[108]:


get_ipython().system('pip install SciPy')
import scipy 


# In[109]:


train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255, zoom_range=0.2,brightness_range=[0.2,1.5], rotation_range=30)
#train_generator = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=[0.2,0.7],brightness_range=[0.2,1.5],width_shift_range=0.2,height_shift_range=0.2, horizontal_flip=False, rotation_range=30,zca_whitening= True, vertical_flip=False)
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
#test_generator = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=[0.2,0.7],brightness_range=[0.2,1.5],width_shift_range=0.2,height_shift_range=0.2, horizontal_flip=False, rotation_range=30,zca_whitening= True,vertical_flip=False)
#train_generator.fit(X)
#test_generator.fit(X_test)
train_generator = train_generator.flow(X_train, y_train, shuffle=False)
test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)


# In[113]:


model_1 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(145, 145, 3)),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(4, activation='softmax') # changed to have 10 neurons (same as number of classes) and 'softmax' activation
])

# Compile the model
model_1.compile(loss="categorical_crossentropy", # changed to categorical_crossentropy
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


# In[115]:


history = model_1.fit(train_generator, epochs=20, validation_data=test_generator, shuffle=True, validation_steps=len(test_generator))


# In[116]:


model_1.save('cropping_face_with_aug.h5')
model_1.save_weights('cropping_face_aug_weights.h5')


# In[117]:


plot_history(history, 'model with only taking face')


# In[118]:


def plot_smoothgrad_of_a_model(model, X):
    score = CategoricalScore([1])

    # Create Saliency visualization object
    saliency = Saliency(model,
                        model_modifier=ReplaceToLinear(),
                        clone=True)

    # Generate saliency map
    saliency_map = saliency(score, X.reshape(-1,145,145,3))
                            #smooth_samples=20) # The number of calculating gradients iterations
                            #smooth_noise=0.20) # noise spread level)

    # Render
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,8))
    #for i, title in enumerate(image_titles):
        #ax[i].set_title(title, fontsize=16)
    ax[0].imshow(X)
    ax[0].set_title('original image')
    ax[1].imshow(saliency_map[0],cmap='jet')
    ax[1].set_title('saliency map')
    fig.suptitle('Original Image vs. Saliency MAP', fontsize = 24)
    #plt.tight_layout()
    plt.show()


# In[119]:




imageDataGenerator_obj = tf.keras.preprocessing.image.ImageDataGenerator( zoom_range=0.2,brightness_range=[0.2,0.3], rotation_range=30)
iterator = imageDataGenerator_obj.flow(X_train, batch_size=1)
plt.figure(figsize = (10,10))
for j in range(6):

    plt.subplot(330 + 1 + j)
    
    chunk = iterator.next()

    sub_img = chunk[0].astype('uint8')

    plt.imshow(sub_img) 
plt.show()


# In[120]:


layer0_output = model_1.layers[2].output
activation_model = tf.keras.Model(inputs = model_1.input, outputs = layer0_output)
img = X_test[59].reshape(-1,145,145,3)
activations = activation_model.predict(img)
plot_featuremaps(img,activations,[model_1.layers[2].name])


# # CONCLUSION :
#     

# 1. WE CAN EFFECTIVELY PREDICT THE CONDITION OF THE PERSONS USING CNN MODELS.
# 2. THE PERFORMANCE OF THE MODEL DEPENDS UPON PRIMARILY ON THE TYPE OF DATA BEING USED TO TRAIN THE MODEL.
# 3. OUR BASE LINE MODEL IS PERFORMING BETTER THAN OTHER MODELS WITH IMAGE AUGMENTATION.THE REASON BEHIND IT MAY BE DUE TO SMALL AMOUNT OF TRAINING DATA AND LACK OF VARIATIONS IN TRAINING DATA.
# 4. BY USING FEATURE MAPS, SALIENCY MAPS,GRAD CAM WE CAN ACTUALLY FAIRLY EXPLAIN OUR MODELS. THE INTERPRETATION OF THE MODELS HELPED US TO UNDERSTAND WHICH ARE THE IMPORTANT FEATURES CNN MODEL IS FOCUSING ON.
# 5. With face cropping our accuracy increases to 92% with augmentation

# In[ ]:




