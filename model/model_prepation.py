
import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D , Dense, Flatten, MaxPooling2D, Dropout
from keras.optimizers import SGD

def inputs_pipeline(directory):
    X, Y = [], []
    img_path = os.path.join(directory)
    for file in os.listdir(img_path):
            image_path =  os.path.join(img_path , file)
            for image in os.listdir(image_path):
                image_directory = os.path.join(image_path, image)
                img = cv2.imread(image_directory)
                face_img = cv2.resize(img, (32, 32))
                X.append(face_img)
                Y.append(file)
    
    X, Y = np.array(X) , np.array(Y)
    Y = to_categorical(Y)
    
    return X, Y


def prep_pixels(train, test):
    """
    We know that the pixel values for each image in the dataset are unsigned integers in the range between no color and full color, or 0 and 255.
    """
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    
    return train_norm, test_norm

def summarize_diagnostics(history):
    """
    These plots are valuable for getting an idea of whether a model is overfitting, underfitting, or has a good fit for the dataset.
    """
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.show()


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # output model
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    # compile model
    opt = SGD(learning_rate = 0.001, momentum = 0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def run_model():

    trainX, trainY = inputs_pipeline("****")
    testX, testY = inputs_pipeline("****")
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model()
    # fit model
    history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=0)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    summarize_diagnostics(history)
    
    if acc> .95:
        model.save('ID_prediction_model.h5')
