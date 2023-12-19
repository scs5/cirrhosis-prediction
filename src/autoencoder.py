from keras.layers import Input, Dense
from keras.models import Model
from utils import *
import numpy as np


def autoencoder():
    num_features = 21

    input_layer = Input(shape=(num_features,))
    encoded = Dense(10, activation='relu')(input_layer)
    decoded = Dense(num_features, activation='sigmoid')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder


if __name__ == '__main__':
    X_train, X_test, y_train = load_data()
    X_train, X_test, y_train, _ = preprocess_data(X_train, X_test, y_train)

    # Combine training and testing data (excluding the 'status' column)
    combined_data = np.concatenate((X_train, X_test), axis=0)

    # Train the autoencoder on the combined data
    ae_model = autoencoder()
    ae_model.fit(combined_data, combined_data, epochs=50, batch_size=32, shuffle=True)
    ae_model.save('./models/autoencoder_model.h5')