from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model

from typing import List

from src.getData import IMG_WIDTH, IMG_HEIGHT

def build_model(base_model:tf.keras.Model=VGG16, dropout:float=.2, fc_layers:List[int]=[1024, 1024], num_classes:int=5) -> tf.keras.Model:
    """
    Build the model
    parameter(s): 
        base_model (tf.keras.Model): base model
        dropout (float): dropout rate
        fc_layers (list): list of units for dense layers
        num_classes (int): number of output of the model
    returns:
        model (tf.keras.Model): model ready to train
    """
    base_model = base_model(include_top=False, weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    trainableLayers = len(base_model.layers) - int(len(base_model.layers) * .2)
    for i, layer in enumerate(base_model.layers):
        if i < trainableLayers:
            layer.trainable = False
        else:
            layer.trainable = True

    x = base_model.output
    x = Flatten()(x)
    for i, fc in enumerate(fc_layers):
        x = Dense(fc, activation='relu', name=f'dense_{i}') (x)
        x = Dropout(dropout)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.output, outputs=predictions)
    return model
