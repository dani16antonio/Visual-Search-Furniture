from typing import List, Tuple, Union, DefaultDict
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
import numpy as pn
from PIL import Image
import os

INPUTPATH = os.path.join('..', 'input')
IMG_WIDTH, IMG_HEIGHT = 224, 224

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
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    return img

def get_image(image_path:tf.Tensor) -> Tuple:
    """
    Load image from the path 
    Parameter(s):
        image_path (tf.Tensor): image path
    retunrs:
        image (tf.Tensor, tf.Tensor): image itself with its label
    """
    image = tf.io.read_file(image_path)
    img = decode_image(image, IMG_WIDTH, IMG_HEIGHT)
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

def get_count_per_class(path:tf.Tensor)-> DefaultDict[str, int]:
    """
    Get a count of items per class
    parameter(s):
        path (tf.Tensor): Tensor with the path of the images.
    returns:
        countDict (dict): dict with the classe as key and its count as value.
    """
    countDict = defaultdict(lambda : 0)
    for imagePath in path:
        imageClass = tf.strings.split(imagePath, sep=os.path.sep)[-2]
        imageClass = imageClass.numpy().decode("ascii")
        countDict[imageClass]+=1
    return countDict

def get_splitted_path(setPath:tf.Tensor, countDict:DefaultDict) -> tf.Tensor:
    """
    Splitting dataset in 2.
    parameter(s):
        setPath (tf.Tensor): path of each image in the dataset
        countDict (DefaultDict): dict with number of image per class for the new set
    return:
        firstSet (tf.Tensor) = principal dataSet
        secondSet (tf.Tensor) = second dataSet
    """
    firstSet, SecondSet = [],[]
    for imagePath in setPath:
        imageClass = tf.strings.split(imagePath, sep=os.path.sep)[-2].numpy().decode("ascii")
        countDict[imageClass] -= 1
        if countDict[imageClass] > 0 :
            SecondSet.append(imagePath)
        else:
            firstSet.append(imagePath)
    firstSet, SecondSet = tf.convert_to_tensor(firstSet, tf.string), tf.convert_to_tensor(SecondSet, tf.string)
    return firstSet, SecondSet

def data_augmentation(path:tf.Tensor) -> tf.Tensor:
    generator = ImageDataGenerator(rotation_range=90,horizontal_flip=True, rescale=1./255)
    dataset = generator.flow_from_directory(path, class_mode='sparse')
    return dataset

def import_data(setType:str, split:bool=False, splitSize:float=.2) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """
    Import image from directory
    Parameter(s):
        setType (str): Name of subdiretory inside of dirPath (train/test/val).
        split (bool): Whether or not to split the dataset.
        splitSize (int): If split equals True, this is the percentage to create another set. 
    Returns:
        dataset (tuple(tf.Tensor(tf.Tensor,tf.Tensor), tf.Tensor(tf.Tensor,tf.Tensor))|tf.Tensor(tf.Tensor,tf.Tensor)): dataset itself with their labels
    """
    
    if split:
        path = os.path.join(INPUTPATH, setType, '*', '*')
        setPath = tf.data.Dataset.list_files(path)
        countDict = get_count_per_class(setPath)
        countDict.update((k, int(v*splitSize)) for k,v in countDict.items())
        firstSetPath, secondSetPath = get_splitted_path(setPath, countDict)

        firstSetPath = tf.data.Dataset.from_tensor_slices(firstSetPath).map(get_image)
        secondSetPath = tf.data.Dataset.from_tensor_slices(secondSetPath).map(get_image)
        return firstSetPath, secondSetPath
    else:
        path = os.path.join(INPUTPATH, setType)
        # setPath = tf.data.Dataset.list_files(path)
        dataset = data_augmentation(path)
        return dataset
