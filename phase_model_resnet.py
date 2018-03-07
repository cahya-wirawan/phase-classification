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
    first_layer = Dense(32)(input)
    block = BatchNormalization()(first_layer)
    block = Activation(activation)(block)
    for i in range(layer_number):
        if i != 0:
            block = BatchNormalization()(block)
            block = Activation(activation)(block)
        block = Dense(32)(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dense(32)(block)
        block = add([first_layer, block])

    block = Activation(activation)(block)
    output = Dense(4, activation='softmax')(block)

    model = Model(inputs=[input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model