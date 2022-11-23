import config
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model



class Classifier(tf.keras.Model):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = Conv2D(32,(3,3))
        self.conv2 = Conv2D(64,(3,3))
        self.bn = BatchNormalization()
        self.relu = Activation('relu')
        self.softmax = Activation('softmax')
        self.pooling = MaxPooling2D((2,2))
        self.flatten = Flatten()
        self.fc1 = Dense(128)
        self.fc2 = Dense(config.NUM_CLASSES)
        self.dropout = Dropout(0.25)
 
    def call(self, images):
        out = self.pooling(self.relu(self.conv1(images)))
        out = self.pooling(self.relu(self.conv2(out)))
        out = self.flatten(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.softmax(self.fc2(out))
        return out
        

def get_model():
    input_layer = Input(shape=config.INPUT_SHAPE)
    classifier = Classifier()(input_layer)
    model = Model(inputs=input_layer, outputs=classifier)

    return model