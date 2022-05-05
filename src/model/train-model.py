import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, RepeatVector
from keras.layers import BatchNormalization
from keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

def load_data(size=3000):

    # make X - images
    for i in range(0, size):
        img_path = 'data/train/images/train_' + str(i) + '.png'
        img = image.load_img(img_path)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        if (i == 0): # if it's the first image, initialize X
            X = x
            continue
        X = np.vstack((X, x))

    # make Y - landmarks
    for i in range(0, size):
        txt_path = 'data/train/landmarks/train_' + str(i) + '.txt'
        with open(txt_path, 'r') as f:
            lines_list = f.readlines()

            for j in range(3, 58): # in landmark text files, landmarks start at 3rd line end in 57th
                string = lines_list[j]
                str1, str2 = string.split(' ')
                x_ = float(str1)
                y_ = float(str2)
                if (j == 3): # if it's the first landmark point, initilialize temp_x, temp_y
                    temp_x = np.array(x_)
                    temp_y = np.array(y_)
                    continue

                # if not first landmark point
                temp_x = np.hstack((temp_x, x_))
                temp_y = np.hstack((temp_y, y_))

        if (i == 0):  # if it's the first image, initialize Y
            Y = np.hstack((temp_x, temp_y))
            Y = Y[None, :]
            continue

        temp = np.hstack((temp_x, temp_y))
        temp = temp[None, :]
        Y = np.vstack((Y, temp))

    return X, Y

def train(X, Y):
    '''Creates a CNN model architecture and fits it to the data given. '''

    model = Sequential()

    # TODO: to change the model architecture, edit below

    ## CNN-MODEL-2.H5 

    model.add(Conv2D(16, (3, 3), input_shape=(224, 224, 3), kernel_initializer='random_uniform', activation='relu'))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(512, (5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))

    model.add(Dense(110))


    ## CNN-MODEL-3.H5 

    # model.add(Conv2D(32, (4, 4), input_shape=(224, 224, 3), activation='relu'))
    # model.add(Conv2D(32, (4, 4), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.1))

    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))

    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

    # model.add(Flatten())
    # model.add(Dense(1500, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1500, activation='relu'))

    # model.add(Dense(110, activation='relu'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=adam, loss="mean_squared_error", metrics=["accuracy"])

    model.fit(X, Y, epochs=500, batch_size=64)

    model.save("../pretrained-models/cnn-models/cnn-model-3.h5") # TODO: change model name here to prevent overwriting

    model.summary()

if __name__ == "__main__":
    # load data
    X, Y = load_data(size=3000)

    # shuffle
    np.random.seed(142)
    np.random.shuffle(X)
    np.random.seed(142)
    np.random.shuffle(Y)

    # train and save model
    train(X, Y) 


