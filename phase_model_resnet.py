from keras.models import Model
from keras.layers import Dropout
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization


# define baseline model
def model_resnet(layers, dropout=0.1, activation='relu', layer_number=10):

    input = Input(shape=(1, 16), name='input')
    first_layer = Dense(64)(input)
    block = Activation(activation)(first_layer)
    for i in range(layer_number):
        block = Dense(64)(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Dense(16)(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = add([input, block])

    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Dense(64)(block)
    block = Activation(activation)(block)
    output = Dense(4, activation='softmax')(block)

    model = Model(inputs=[input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model