#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install tensorflow==2.9.1')


# In[ ]:


# To have reproducible results and compare them
nr_seed = 2019
import numpy as np
np.random.seed(nr_seed)
import tensorflow as tf
print(tf.__version__) # we want 2.9.1
tf.random.set_seed(nr_seed)


# In[ ]:


tf.compat.v1.disable_eager_execution()


# In[ ]:


# import libraries
import json
import math
from tqdm import tqdm, tqdm_notebook
import gc
import warnings
import os

import cv2
from PIL import Image

import pandas as pd
import scipy
import matplotlib.pyplot as plt

from keras import backend as K
from keras import layers
from keras.applications.densenet import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score

warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


im_size = 320
BATCH_SIZE = 32


# In[ ]:


get_ipython().system(" unzip '/content/drive/MyDrive/Colab Notebooks/Explainability/APTOS2019/APTOS2019.zip'")


# In[ ]:


train_df= pd.read_csv('/content/train_1.csv')
val_df= pd.read_csv('/content/valid.csv')
test_df= pd.read_csv('/content/test.csv')
train_df.head()


# In[ ]:


import numpy as np
#As you can see,the data is imbalanced.
#So we've to calculate weights for each class,which can be used in calculating loss.

from sklearn.utils import class_weight #For calculating weights for each class.
class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.array([0,1,2,3,4]),y=train_df['diagnosis'].values)
# Convert class weights to a TensorFlow tensor
class_weights= tf.convert_to_tensor(class_weights, dtype=tf.float32)


print(class_weights) 


# In[ ]:


train_df = train_df.sample(frac=1).reset_index(drop=True)
val_df = val_df.sample(frac=1).reset_index(drop=True)
print(train_df.shape)
print(val_df.shape)


# In[ ]:


# Crop function: https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance

    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def circle_crop(img):
    img = crop_image(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = width//2
    y = height//2
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image(img)

    return img

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def preprocess_image(image_path, desired_size=448):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img= circle_crop(img)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size,desired_size))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), desired_size/30) ,-4 ,128)

    return img

def preprocess_image_old(image_path, desired_size=448):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img= circle_crop(img)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size,desired_size))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), desired_size/40) ,-4 ,128)

    return img


# In[ ]:


def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_name = df.loc[i,'id_code'] #c947bb6cf9f6

        image_id = df.loc[i,'diagnosis']

        #img = cv2.imread(f'{image_path}')
        image_name= image_name+'.png'

        img_path= os.path.join('/content/train_images/train_images/', image_name)

        img= cv2.imread(img_path)
        #img= cv2.imread('/content/train_images/train_images')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_image_from_gray(img)
        img = cv2.resize(img, (im_size,im_size))
        img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), im_size/40) ,-4 ,128)

        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)

    plt.tight_layout()

display_samples(train_df)


# In[ ]:


# validation set
N = val_df.shape[0]
print(N)
x_val = np.empty((N, im_size, im_size, 3), dtype=np.uint8) #(366, 320, 320, 3)

for i, image_id in enumerate(tqdm_notebook(val_df['id_code'])):

    image_id= image_id+'.png'
    img_path= os.path.join('/content/val_images/val_images', image_id)
    x_val[i, :, :, :] = preprocess_image(
        img_path,
        desired_size = im_size
    )
print(type(x_val))

#training set
N = train_df.shape[0]
print(N)
x_train = np.empty((N, im_size, im_size, 3), dtype=np.uint8) #(366, 320, 320, 3)

for i, image_id in enumerate(tqdm_notebook(train_df['id_code'])):
    #print(i)
    image_id= image_id+'.png'
    img_path= os.path.join('/content/train_images/train_images', image_id)
    x_train[i, :, :, :] = preprocess_image(
        img_path,
        desired_size = im_size
    )



# In[ ]:


y_train = pd.get_dummies(train_df['diagnosis']).values
y_val = pd.get_dummies(val_df['diagnosis']).values
y_test = pd.get_dummies(test_df['diagnosis']).values
#print(x_train.shape)
print(y_train.shape)
#print(x_val.shape)
print(y_val.shape)


# In[ ]:


# Convert one-hot encoded labels to integer format
y_train_labels = np.argmax(y_train, axis=1)
print(y_train_labels)


# In[ ]:


#https://medium.com/swlh/multi-class-classification-with-focal-loss-for-imbalanced-datasets-c478700e65f5
def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002
        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]
        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})
        Returns:
            [tensor] -- loss.
        """

        epsilon = 1.e-9
        #print("Y_true",y_true) #Y_true Tensor("IteratorGetNext:1", shape=(None, None), dtype=uint8)
        #print("Y_pred",y_pred) #Y_pred Tensor("model_1/dense_1/Softmax:0", shape=(None, 5), dtype=float32)
        y_true= tf.cast(y_true, dtype=tf.float32)
        #y_true = tf.convert_to_tensor(y_true, tf.float32)

        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)

        ce = tf.multiply(y_true, -tf.math.log(model_out))

        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))

        fl = tf.multiply(alpha, tf.multiply(weight, ce))

        reduced_fl = tf.reduce_max(fl, axis=1)

        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed


# In[ ]:


#focal loss for class specific alpha and gamma
import tensorflow as tf
from tensorflow.keras import backend as K

def focal_loss(alpha, gamma):
    def loss_fn(y_true, y_pred):
        # Convert integer labels to one-hot encoded labels
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

        # Calculate probabilities from logits
        y_pred_prob = tf.nn.softmax(y_pred, axis=-1)

        # Apply alpha values to one-hot encoded labels
        alpha_factor = tf.reduce_sum(alpha * y_true_one_hot, axis=-1)

        # Calculate focal weights
        focal_weights = tf.pow(1 - y_pred_prob, gamma)

        # Calculate cross-entropy loss with focal weights and alpha values
        ce_loss = -alpha_factor * focal_weights * tf.math.log(y_pred_prob)

        # Calculate mean loss
        focal_loss = tf.reduce_mean(ce_loss, axis=-1)

        return focal_loss

    return loss_fn


# In[ ]:


import tensorflow as tf

def focal_loss(alpha, gamma):
    def loss_fn(y_true, y_pred):
        # Convert integer labels to one-hot encoded labels
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

        # Calculate cross-entropy loss
        ce_loss = tf.losses.CategoricalCrossentropy(from_logits=True)(y_true_one_hot, y_pred)

        # Calculate focal loss
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)
        alpha_factor = tf.reduce_sum(alpha_tensor * y_true_one_hot, axis=1)
        pt = tf.exp(-ce_loss)
        focal_loss = alpha_factor * tf.pow(1.0 - pt, gamma) * ce_loss

        return focal_loss

    return loss_fn


# In[ ]:


def focal_reweighted_loss(gamma=2.0, alpha=0.25):

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)  # Clip values to prevent division by zero

        # Compute focal loss
        focal_loss = -alpha * (1 - y_pred) ** gamma * y_true * tf.math.log(y_pred)

        # Compute reweighting factor
        class_weights = tf.reduce_sum(y_true, axis=0) / tf.reduce_sum(y_true)
        weighted_loss = tf.reduce_sum(class_weights * focal_loss)

        return weighted_loss

    return loss_fn


# In[ ]:


#dice loss
def dice_loss(y_true, y_pred):
    smooth = 1e-5  # smoothing parameter to prevent division by zero
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    union = tf.reduce_sum(y_true + y_pred, axis=-1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice

    return loss


# In[ ]:


#dice loss + CE
from tensorflow.keras.losses import categorical_crossentropy
def combined_loss(y_true, y_pred, dice_weight=0.6, epsilon=1e-7):
    # Compute the Dice Loss
    dice = dice_loss(y_true, y_pred)

    # Compute the weighted cross-entropy loss
    cross_entropy = categorical_crossentropy(y_true, y_pred)

    # Combine the two losses
    combined_loss = dice_weight * dice + (1 - dice_weight) * cross_entropy
    return combined_loss


# In[ ]:


#dice loss+ weighted CE
def weighted_cross_entropy_loss(class_weights):
    class_weights = K.constant(class_weights)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Apply weights to each class
        weighted_losses = y_true * K.log(y_pred) * class_weights
        # Sum the losses across all classes
        loss = -K.sum(weighted_losses, axis=-1)
        return loss

    return loss

def combined_loss(dice_weight, class_weights):
    dice = dice_loss
    weighted_ce = weighted_cross_entropy_loss(class_weights)

    def loss(y_true, y_pred):
        # Compute dice loss
        dice_loss = dice(y_true, y_pred)
        # Compute weighted cross entropy loss
        ce_loss = weighted_ce(y_true, y_pred)
        # Combine the losses
        combined_loss = dice_weight * dice_loss + (1 - dice_weight) * ce_loss
        return combined_loss

    return loss


# In[ ]:


#Dice loss+ Focal loss
def combined_loss(alpha=1, gamma=2,dice_weight=0.5):
    dice = dice_loss
    focal = focal_loss(alpha, gamma)

    def loss(y_true, y_pred):
        dice_loss = dice(y_true, y_pred)
        focal_loss = focal(y_true, y_pred)
        combined_loss = dice_weight * dice_loss + (1 - dice_weight) * focal_loss
        return combined_loss

    return loss


# In[ ]:


#Dice loss+ WCE+ Focal loss
def weighted_cross_entropy_loss(class_weights):
    class_weights = K.constant(class_weights)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Apply weights to each class
        weighted_losses = y_true * K.log(y_pred) * class_weights
        # Sum the losses across all classes
        loss = -K.sum(weighted_losses, axis=-1)
        return loss

    return loss

def combined_loss(dice_weight, CE_weight,  class_weights, alpha,gamma):
    dice = dice_loss
    weighted_ce = weighted_cross_entropy_loss(class_weights)
    focal= focal_loss(gamma, alpha)

    def loss(y_true, y_pred):
        # Compute dice loss
        dice_loss = dice(y_true, y_pred)
        # Compute weighted cross entropy loss
        ce_loss = weighted_ce(y_true, y_pred)
        focal_loss= focal(y_true, y_pred)
        # Combine the losses
        combined_loss = dice_weight * dice_loss + CE_weight* ce_loss + (1-dice_weight-CE_weight)*focal_loss
        return combined_loss

    return loss



# In[ ]:


#lovasz_softmax_loss + WCE+ focal loss
def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002
        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]
        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})
        Returns:
            [tensor] -- loss.
        """

        epsilon = 1.e-9
        #print("Y_true",y_true) #Y_true Tensor("IteratorGetNext:1", shape=(None, None), dtype=uint8)
        #print("Y_pred",y_pred) #Y_pred Tensor("model_1/dense_1/Softmax:0", shape=(None, 5), dtype=float32)
        y_true= tf.cast(y_true, dtype=tf.float32)
        #y_true = tf.convert_to_tensor(y_true, tf.float32)

        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)

        ce = tf.multiply(y_true, -tf.math.log(model_out))

        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))

        fl = tf.multiply(alpha, tf.multiply(weight, ce))

        reduced_fl = tf.reduce_max(fl, axis=1)

        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

def weighted_cross_entropy_loss(class_weights):
    class_weights = K.constant(class_weights)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Apply weights to each class
        weighted_losses = y_true * K.log(y_pred) * class_weights
        # Sum the losses across all classes
        loss = -K.sum(weighted_losses, axis=-1)
        return loss

    return loss

def combined_loss(lovasz_weight, CE_weight,  class_weights, alpha,gamma):

    lovasz=lovasz_softmax_loss
    weighted_ce = weighted_cross_entropy_loss(class_weights)
    focal= focal_loss(gamma, alpha)

    def loss(y_true, y_pred):
        # Compute dice loss
        lovasz_softmax_loss = lovasz(y_true, y_pred)
        # Compute weighted cross entropy loss
        ce_loss = weighted_ce(y_true, y_pred)
        focal_loss= focal(y_true, y_pred)
        # Combine the losses
        combined_loss = lovasz_weight * lovasz_softmax_loss+ CE_weight* ce_loss + (1-lovasz_weight -CE_weight)*focal_loss
        return combined_loss

    return loss



# In[ ]:


import keras.backend as K

def class_tversky(y_true, y_pred):
    smooth = 1

    #y_true = K.permute_dimensions(y_true, (3,1,2,0))
    #y_pred = K.permute_dimensions(y_pred, (3,1,2,0))
    y_true = tf.cast(y_true, tf.float32)
    y_true_pos = K.batch_flatten(y_true)
    y_pred_pos = K.batch_flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos, 1)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
    false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
    alpha = 0.6
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def focal_tversky_loss(y_true,y_pred):
    pt_1 = class_tversky(y_true, y_pred)
    gamma =1.5
    return K.sum(K.pow((1-pt_1), gamma))


# In[ ]:


#focal tversky+ CE
from tensorflow.keras.losses import categorical_crossentropy
def combined_loss(ce_weight=0.5):
    def loss_func(y_true, y_pred):
        # Calculate Cross Entropy Loss
        ce_loss = categorical_crossentropy(y_true, y_pred)
        # Calculate Focal Tversky Loss
        ft_loss = focal_tversky_loss(y_true, y_pred)

        # Combine the losses
        combined_loss = ce_weight * ce_loss + (1 - ce_weight) * ft_loss

        return combined_loss

    return loss_func


# In[ ]:


#focal tversky+ WCE
from tensorflow.keras.losses import categorical_crossentropy

def weighted_cross_entropy_loss(class_weights):
    class_weights = K.constant(class_weights)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Apply weights to each class
        weighted_losses = y_true * K.log(y_pred) * class_weights
        # Sum the losses across all classes
        loss = -K.sum(weighted_losses, axis=-1)
        return loss

    return loss


def combined_loss(ce_weight=0.5):

    weighted_ce = weighted_cross_entropy_loss(class_weights)
    def loss_func(y_true, y_pred):
        # Calculate Cross Entropy Loss
        ce_loss = weighted_ce(y_true, y_pred)
        # Calculate Focal Tversky Loss
        ft_loss = focal_tversky_loss(y_true, y_pred)

        # Combine the losses
        combined_loss = ce_weight * ce_loss + (1 - ce_weight) * ft_loss

        return combined_loss

    return loss_func


# In[ ]:


#Focal twersky + WCE + dice

def weighted_cross_entropy_loss(class_weights):
    class_weights = K.constant(class_weights)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Apply weights to each class
        weighted_losses = y_true * K.log(y_pred) * class_weights
        # Sum the losses across all classes
        loss = -K.sum(weighted_losses, axis=-1)
        return loss

    return loss
#dice loss
def dice_loss(y_true, y_pred):
    smooth = 1e-5  # smoothing parameter to prevent division by zero
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    union = tf.reduce_sum(y_true + y_pred, axis=-1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice

    return loss

def combined_loss(CE_weight,dice_weight, class_weights):
    dice = dice_loss
    weighted_ce = weighted_cross_entropy_loss(class_weights)


    def loss_func(y_true, y_pred):
        # Calculate Cross Entropy Loss
        ce_loss = weighted_ce(y_true, y_pred)
        # Calculate Focal Tversky Loss
        ft_loss = focal_tversky_loss(y_true, y_pred)
        dice_loss = dice(y_true, y_pred)
        # Combine the losses
        #combined_loss = ce_weight * ce_loss + (1 - ce_weight) * ft_loss
        combined_loss = dice_weight * dice_loss + CE_weight* ce_loss + (1-dice_weight-CE_weight)*ft_loss
        return combined_loss

    return loss_func



# In[ ]:


#focal tversky+dice
#dice loss
def dice_loss(y_true, y_pred):
    smooth = 1e-5  # smoothing parameter to prevent division by zero
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    union = tf.reduce_sum(y_true + y_pred, axis=-1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice

    return loss

def combined_loss(dice_weight):
    dice = dice_loss


    def loss_func(y_true, y_pred):

        # Calculate Focal Tversky Loss
        ft_loss = focal_tversky_loss(y_true, y_pred)
        dice_loss = dice(y_true, y_pred)
        # Combine the losses
        #combined_loss = ce_weight * ce_loss + (1 - ce_weight) * ft_loss
        combined_loss = dice_weight * dice_loss +  (1-dice_weight)*ft_loss
        return combined_loss

    return loss_func




# In[ ]:


class Metrics(Callback):
    def __init__(self, val_data, batch_size = 32):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size
        self.save_path = '/content/best_path/'
        self.best_kappa = -1
        self.val_kappas=[]
    def on_epoch_end(self, epoch, logs={}):

        #X_val, y_val = self.validation_data[:2]
        xVal, yVal = next(self.validation_data)
        #y_val = y_val.sum(axis=1) - 1
        val_pred = []
        val_true = []

        xVal = tf.cast(xVal, dtype=tf.float32)
        y_pred= self.model.predict(xVal) # probabilities

        val_pred.extend(tf.argmax(y_pred, axis=1)) #[<tf.Tensor: shape=(), dtype=int64, numpy=0>
        print(val_pred)
        yVal = tf.cast(yVal, dtype=tf.float32)
        val_true.extend(tf.argmax(yVal, axis=1)) #[<tf.Tensor: shape=(), dtype=int64, numpy=0>
        print(val_true)


        print("EPooooch")
        print(epoch)

        _val_kappa = cohen_kappa_score(
            val_true,
            val_pred,
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")

        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return


# In[ ]:


# data generator

def create_datagen():
    return ImageDataGenerator(
        featurewise_std_normalization = True,
        horizontal_flip = True,
        vertical_flip = True,
        rotation_range = 360
    )


# In[ ]:


# functional api
from keras.applications import EfficientNetB0, DenseNet121,ResNet50V2,VGG16,InceptionResNetV2,Xception,ResNet50,VGG16,InceptionV3,MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model

# Load the EfficientNet-B0 model (excluding the top classifier)
base_model = Xception(weights='imagenet', include_top=False, input_shape=(320, 320, 3))

# Freeze the base model's layers
#base_model.trainable = False
'''
for layer in base_model.layers:
        layer.trainable = False

for i in range(-110, 0):
        base_model.layers[i].trainable = True
'''
# Create the inputs and pass them through the base model
#inputs = tf.keras.Input(shape=(320, 320, 3))
#pretrained_outputs = base_model(inputs)
x = base_model.output

# Add a GlobalAveragePooling2D layer
x = GlobalAveragePooling2D()(x)
x= Dropout(0.5)(x)
# Add a fully connected layer with the desired number of classes
outputs = Dense(5, activation='softmax')(x)

# Create the finetuned model
model = Model(base_model.input, outputs)


# In[ ]:


# Define your custom loss function
#@tf.function
def weighted_loss(y_true, y_pred):
    # Convert y_true to float32
    y_true = tf.cast(y_true, tf.float32)

    # Calculate the weighted cross-entropy loss
    loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, class_weights)

    # Calculate the mean loss over the batch
    loss = tf.reduce_mean(loss)
    return loss


# In[ ]:


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
#CE
#model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001,decay=1e-6), metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
#Weighted loss
#model.compile(loss=weighted_loss,optimizer=Adam(lr=0.0001,decay=1e-6),metrics=['accuracy'])
#model.compile(loss=weighted_loss,optimizer=optimizer,metrics=['accuracy'])

#Focal loss
#model.compile(optimizer=Adam(lr=0.0001,decay=1e-6), loss=focal_loss(gamma=2.5,alpha=1), metrics=['accuracy'])
#model.compile(loss=focal_loss(gamma=2.5,alpha=1.5),optimizer=optimizer,metrics=['accuracy'])
#model.compile(optimizer=Adam(lr=0.0001,decay=1e-6), loss=focal_loss(alpha=alpha, gamma=gamma), metrics=['accuracy'])
#Dice loss
#model.compile(optimizer=Adam(lr=0.0001,decay=1e-6), loss=dice_loss, metrics=['accuracy'])
#model.compile(optimizer=optimizer, loss=dice_loss, metrics=['accuracy'])
#Dice + CE
#model.compile(optimizer=Adam(lr=0.0001,decay=1e-6), loss=combined_loss, metrics=['accuracy'])
#Dice + weighted CE
model.compile(optimizer=optimizer, loss=combined_loss(dice_weight=0.4 ,class_weights=class_weights), metrics=['accuracy'])
#Dice + Focal loss
#model.compile(optimizer=optimizer, loss=combined_loss(alpha=0.25 ,gamma=2,dice_weight=0.5), metrics=['accuracy'])
#Dice+WCE+Focal loss
#model.compile(optimizer=optimizer, loss=combined_loss(alpha=1,gamma=2,dice_weight=0.2,class_weights=class_weights,CE_weight=0.4), metrics=['accuracy'])


#KLD loss
#model.compile(optimizer=Adam(lr=0.0001,decay=1e-6), loss=kld_loss, metrics=['accuracy'])

#Focal loss+ reweighing
#model.compile(optimizer=Adam(lr=0.0001,decay=1e-6), loss=focal_reweighted_loss(alpha=1), metrics=['accuracy'])

#Focal twersky loss
#model.compile(optimizer=optimizer, loss=focal_tversky_loss, metrics=['accuracy'])

#Focal tversky + CE/ #Focal tversky + WCE
#model.compile(optimizer=Adam(lr=0.0001,decay=1e-6), loss=combined_loss(ce_weight=0.5), metrics=['accuracy'])
#model.compile(optimizer=optimizer, loss=combined_loss(ce_weight=0.5), metrics=['accuracy'])
#Focal tversky + dice
#model.compile(optimizer=Adam(lr=0.0001,decay=1e-6), loss=combined_loss( dice_weight= 0.5 ), metrics=['accuracy'])
#Focal tversky + WCE+ dice
#model.compile(optimizer=Adam(lr=0.0001,decay=1e-6), loss=combined_loss(CE_weight=0.3, dice_weight= 0.4,class_weights=class_weights ), metrics=['accuracy'])


#lovasz_softmax_loss
#model.compile(optimizer=Adam(lr=0.0001,decay=1e-6), loss=lovasz_softmax_loss, metrics=['accuracy'])
#lovasz_softmax_loss+ WCE
#model.compile(optimizer=Adam(lr=0.0001,decay=1e-6), loss=combined_loss(CE_weight=0.2, class_weights=class_weights ), metrics=['accuracy'])
#lovasz_softmax_loss+ WCE+ Focal loss
#model.compile(optimizer=Adam(lr=0.0001,decay=1e-6), loss=combined_loss(lovasz_weight=0.4,CE_weight=0.3,
                                                                      # class_weights=class_weights,alpha=1,gamma=2 ), metrics=['accuracy'])
#lovasz_weight, CE_weight,  class_weights, alpha,gamma



model.summary()


# In[ ]:


from keras.utils import plot_model
# Plot the model figure
plot_model(model, to_file='model_figure.png', show_shapes=True)


# In[ ]:


#train_df = train_df.reset_index(drop=True)
bucket_num = 8
div = round(x_train.shape[0]/bucket_num)
print(div) #samples per bucket


# In[ ]:


df_init = {
    'val_loss': [0.0],
    'val_acc': [0.0],
    'loss': [0.0],
    'acc': [0.0],
    'bucket': [0.0]
}
results = pd.DataFrame(df_init)


# In[ ]:


val_data_generator= create_datagen().flow(x_val,y_val,batch_size=BATCH_SIZE )
print("Val data")
print(val_data_generator)
for i, (X_batch, y_batch) in enumerate(val_data_generator):
            print(f"Batch {i+1} - X shape: {X_batch.shape}, y shape: {y_batch.shape}")

    # Break the loop after a few batches to avoid printing the entire generator
            if i == 2:
                break


# In[ ]:


# I found that changing the nr. of epochs for each bucket helped in terms of performances

epochs = [5,5,10,15,15,20,20,30]
#epochs=[5,5]
#epochs = [5,5,5,5,5,5,5,5]
#epochs=[10,10,10,10]

kappa_metrics = Metrics(val_data_generator)


# In[ ]:


for i in range(0,bucket_num):

    if i != (bucket_num-1):
        print("Bucket Nr: {}".format(i))

        N = train_df.iloc[i*div:(1+i)*div].shape[0]
        #train_df.iloc[i*div:(1+i)*div] create subset of samples in a range
        #print(N) #366
        x_train = np.empty((N, im_size, im_size, 3), dtype=np.uint8)
        for j, image_id in enumerate(tqdm_notebook(train_df.iloc[i*div:(1+i)*div,0])):#It selects the values from the first column of the DataFrame

            image_id= image_id+'.png'
            img_path= os.path.join('/content/train_images/train_images', image_id)

            x_train[j, :, :, :] = preprocess_image(img_path, desired_size = im_size)

        data_generator = create_datagen().flow(x_train, y_train[i*div:(1+i)*div,:], batch_size=BATCH_SIZE)
        print(data_generator)
        # Iterate over the generator and print the shape of each batch
        for k, (X_batch, y_batch) in enumerate(data_generator):
            print(f"Batch {k+1} - X shape: {X_batch.shape}, y shape: {y_batch.shape}")

    # Break the loop after a few batches to avoid printing the entire generator
            if k == 4:
                break
        history = model.fit(
                        data_generator,
                        steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
                        epochs=epochs[i], validation_data= val_data_generator,
                        callbacks=[kappa_metrics]
                        )

        dic = history.history
        df_model = pd.DataFrame(dic)
        df_model['bucket'] = i
    else:
        print("Bucket Nr: {}".format(i))

        N = train_df.iloc[i*div:].shape[0]
        print(N, i*div)
        x_train = np.empty((N, im_size, im_size, 3), dtype=np.uint8)
        for j, image_id in enumerate(tqdm_notebook(train_df.iloc[i*div:,0])):

            image_id= image_id+'.png'
            img_path= os.path.join('/content/train_images/train_images', image_id)

            x_train[j, :, :, :] = preprocess_image(img_path, desired_size = im_size)
        data_generator = create_datagen().flow(x_train, y_train[i*div:,:], batch_size=BATCH_SIZE)

        history = model.fit(
                        data_generator,
                        steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
                        epochs=epochs[i], validation_data= val_data_generator,
                        callbacks=[kappa_metrics]
                        )

        dic = history.history
        df_model = pd.DataFrame(dic)
        df_model['bucket'] = i

    results = results.append(df_model)

    #del data_generator
    #del x_train
    gc.collect()

    print('-'*40)


# In[ ]:


results = results.iloc[1:]
results['kappa'] = kappa_metrics.val_kappas
results = results.reset_index()
results = results.rename(index=str, columns={"index": "epoch"})
results


# In[ ]:


results[['loss', 'val_loss']].plot()
results[['accuracy', 'val_accuracy']].plot()
results[['kappa']].plot()
results.to_csv('model_results.csv',index=False)


# In[ ]:


# Predicting on test
import tensorflow as tf
from tensorflow.keras import utils
import pandas as pd
import numpy as np
#from tensorflow.keras.models import load_model
from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score,f1_score


# when you use weighted loss function, run the next two code statements, rather than going directly to load_model
# Register the custom loss function
# Register the custom loss function
#custom_objects = {'focal_loss_fixed': focal_loss()} #focal loss
#custom_objects = {'loss_fn': focal_loss()}
#custom_objects = {'loss_fn': focal_reweighted_loss()} #focal_reweighted loss
#custom_objects = {'loss': combined_loss(alpha=0.25,gamma=2,dice_weight=0.4)} #Dice+ Focal loss
custom_objects = {'loss': combined_loss(dice_weight=0.4,class_weights=class_weights)} #Dice+ WCE
#custom_objects = {'loss': combined_loss(alpha=0.25,gamma=2,dice_weight=0.2,CE_weight=0.5,class_weights=class_weights)} #Dice+#WCE+Focal
#custom_objects = {'loss_func': combined_loss(CE_weight=0.3,dice_weight=0.4,class_weights=class_weights)} #Dice + WCE+ Focal tversky
#custom_objects = {'loss': combined_loss(CE_weight=0.2,class_weights=class_weights)} #lovasz_softmax_loss+ WCE
#custom_objects = {'loss': combined_loss(lovasz_weight=0.4,CE_weight=0.3,class_weights=class_weights,alpha=1,gamma=2)} #lovasz_softmax+ WCE+ Focal loss



#custom_objects = {'loss_func': combined_loss(dice_weight=0.5)} #Dice +Focal tversky
#custom_objects = {'loss_func': combined_loss(ce_weight=0.5)} # focal tversky+ CE
#tf.keras.utils.get_custom_objects()['weighted_loss'] = weighted_loss
#tf.keras.utils.get_custom_objects()['dice_loss'] =dice_loss
#tf.keras.utils.get_custom_objects()['kld_loss'] =kld_loss
#tf.keras.utils.get_custom_objects()['focal_tversky_loss'] =focal_tversky_loss ##focal_tversky_loss
#tf.keras.utils.get_custom_objects()['lovasz_softmax_loss'] =lovasz_softmax_loss
#lovasz_softmax_loss

# Load your model
#model = tf.keras.models.load_model('/content/model.h5', custom_objects={'weighted_loss': weighted_loss})
#model = tf.keras.models.load_model('/content/model.h5', custom_objects={'dice_loss': dice_loss})
#model = tf.keras.models.load_model('/content/model.h5', custom_objects={'combined_loss': combined_loss}) #Dice + CE
#model = tf.keras.models.load_model('/content/model.h5', custom_objects={'kld_loss': kld_loss})



model = tf.keras.models.load_model('/content/model.h5', custom_objects=custom_objects) #focal loss and focal reweighted, dice+WCE, dice+focal
# Load the trained model (normal loss function, without weighted loss)
#model = load_model('/content/drive/MyDrive/DR_Files/best_model_WCE_1.h5')  # Replace 'model.h5' with the actual file path to your trained model
#model = tf.keras.models.load_model('/content/model.h5')
# Load the test data from CSV file
test_data = pd.read_csv('/content/test.csv')  # Replace 'test_data.csv' with the actual file path to your test data

# Preprocess the test data (e.g., resize images, normalize pixel values, etc.)
# ...

# Extract the image data from the test data
test_images = test_data['id_code'].values  # Replace 'image_path' with the actual column name containing the image paths
test_labels= test_data['diagnosis'].values
# Create an empty list to store the predicted labels
predicted_labels = []
true_labels = []
# Iterate over the test images and make predictions
for image_id,label  in zip(test_images,test_labels):
    # Load and preprocess the image
    image_id= image_id+'.png'
    image_path= os.path.join('/content/test_images/test_images', image_id)
    image = preprocess_image(image_path, desired_size = im_size)  # Replace 'preprocess_image' with your own image preprocessing function

    # Expand dimensions to match the input shape of the model
    image = np.expand_dims(image, axis=0)

    # Make the prediction
    prediction = model.predict(image)

    # Get the predicted label (e.g., by taking the argmax if it's a classification task)
    predicted_label = np.argmax(prediction)

    # Append the predicted label to the list
    predicted_labels.append(predicted_label)
    true_labels.append(label)

# Convert the list of predicted labels to a NumPy array
predicted_labels = np.array(predicted_labels)
true_labels = np.array(true_labels)



kappa = cohen_kappa_score(true_labels, predicted_labels, weights='quadratic')
print('Quadratic Kappa Score:', kappa)

# Calculate precision, recall, and F1 score
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print('Precision:(weighted)', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# Calculate the confusion matrix
confusion_matrix = tf.math.confusion_matrix(true_labels, predicted_labels)

# Convert the confusion matrix to a NumPy array for printing
confusion_matrix = confusion_matrix.numpy()

# Print the confusion matrix
print(confusion_matrix)
cm= confusion_matrix
num_classes = cm.shape[0]
TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP
TN = np.sum(cm) - (TP + FP + FN)

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print('Sensitivity:', sensitivity)
print('Specificity:',specificity)


# In[ ]:




