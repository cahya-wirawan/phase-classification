from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense


# define baseline model
def model_simple(layers, dropout=0.1, layer_number=None):
    # create model
    model = Sequential()
    model.add(Dense(layers[0], input_shape=(1, 16), activation='relu'))
    model.add(Dropout(dropout))
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(4, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model