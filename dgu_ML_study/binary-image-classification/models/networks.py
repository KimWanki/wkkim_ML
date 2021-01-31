from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, Activation, MaxPooling2D

class CNN:
    def __init__(self, size):
        self.size=size
    def build(self):   
        size=self.size
        print('size: ', size)
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(size, size, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Activation('sigmoid'))
        return model