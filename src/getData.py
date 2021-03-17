from typing import List, Tuple

import tensorflow as tf
import numpy as pn
from PIL import Image
import os

INPUTPATH = os.path.join('..', 'input')

def get_label(image_path:str)->str:
    """
    get image label from the path
    Parameter(s):
        image_path (str): image path
    retunrs:
        label (str): label of the image
    """
    parts = tf.strings.split(image_path, os.path.sep)
    label = parts[-2]
    return label

#tf.python.framework.ops.EagerTensor
def decode_image(img:tf.Tensor, img_width:int, img_height:int)->tf.Tensor:
    """
    convert Tensor of string to tensor of float32 and add 3 channels.
    parameter(s):
        img (tf.python.framework.ops.EagerTensor) tensor with the image data.
        img_width (int): width to rescale the image.
        img_height (int): height to rescale the image.
    returns:
        img (tf.Tensor): image tensor
    """
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_width, img_height])
    return img

def get_image(image_path:str) -> Tuple:
    """
    Load image from the path 
    Parameter(s):
        image_path (str): image path
        img_width (int): width to rescale the image
        img_height (int): height to rescale the image
    retunrs:
        image (tf.Tensor, tf.Tensor): image itself with its label
    """
    img_width,img_height = 224, 224
    image = tf.io.read_file(image_path)
    img = decode_image(image, img_width, img_height)
    label = get_label(image_path)
    return img, label

def get_classes() -> List:
    """
    Get all clases from the image path.
    Parameter(s):

    Return(s):
        classes (list): all the classes.
    """
    setPath = os.listdir(os.path.join(INPUTPATH, 'train'))
    classes = []
    for cl in setPath:
        cl = tf.strings.split(cl, os.path.sep)[-1]
        if not cl in classes:
            classes.append(cl)
    return classes

def import_data(setType:str, split:bool=False, splitSize:int=None) -> tf.Tensor:
    """
    Import image from directory
    Parameter(s):
        setType (str): Name of subdiretory inside of dirPath (train/test/val).
        split (bool): Whether or not to split the dataset.
        splitSize (int): If split equals True, this is the percentage of split to create another set. 
    Returns:
        dataset (tuple(tf.Tensor(tf.Tensor,tf.Tensor), tf.Tensor(tf.Tensor,tf.Tensor))|tf.Tensor(tf.Tensor,tf.Tensor)): dataset itself with their labels
    """
    path = os.path.join(INPUTPATH, setType, '*', '*')
    setPath = tf.data.Dataset.list_files(path)
    dataset = setPath.map(get_image)
    
    return dataset
