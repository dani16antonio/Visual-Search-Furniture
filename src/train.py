from tensorflow.keras.optimizers import Adam

from getData import import_data
from model import build_model

def train_model():
    trainSet = None
    valSet, testSet = None, None
    EPOCHS = 20

    # Get data
    trainSet = import_data("train")
    valSet, testSet = import_data("val", True, .2)

    # get and compile the model
    model = build_model()
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    # train the model

    history = model.fit(trainSet, epochs=EPOCHS, validation_data=valSet)

    return model

if __name__ == '__main__':
    train_model()
