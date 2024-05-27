from tensorflow.keras.layers import Input, Conv1D, Dropout, Conv1DTranspose, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

class Conv_RNN_AE(Model): 
    """
    A reconstruction convolutional+recurrent autoencoder model to detect anomalies in timeseries data using reconstruction error as an anomaly score.

    Parameters
    ----------
    dilation_rates : list
        List of dilation rates for the convolutional layers.
    filters : list
        List of filter sizes for the convolutional layers.
    loss : str
        Loss function to use for training the model.

    Attributes
    ----------
    model : Model
        The trained convolutional autoencoder model.

    Examples
    --------
    >>> from Conv_AE import Conv_AE
    >>> CAutoencoder = Conv_AE()
    >>> CAutoencoder.fit(train_data)
    >>> prediction = CAutoencoder.predict(test_data)
    """
    def _Random(self, seed_value): 
        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        import random
        random.seed(seed_value)

        import numpy as np
        np.random.seed(seed_value)

        import tensorflow as tf
        tf.random.set_seed(seed_value)

    def __init__(
        self,
        dilation_rates=[2, 4, 8],
        filters=[32, 16, 8],
        loss="mse"
        ):
        super(Conv_RNN_AE, self).__init__()
        self._Random(0)
        self.dilation_rates = dilation_rates
        self.filters = filters
        self.loss = loss

        self.encoder_layers = []
        self.decoder_layers = []

        # Encoder with progressive dilation rates and filters
        for dilation_rate, filters in zip(self.dilation_rates, self.filters):
            self.encoder_layers.append(
                Conv1D(
                    filters=filters, kernel_size=3, dilation_rate=dilation_rate, padding="same", activation="relu"
                )
            )
            self.encoder_layers.append(Dropout(rate=0.2))
        
        # GRU layer
        self.gru_layer = GRU(units=8, return_sequences=True)  # Adjust units as needed

        # Decoder with decreasing dilation rates and filters
        for dilation_rate, filters in zip(self.dilation_rates[::-1], self.filters[::-1]):
            self.decoder_layers.append(
                Conv1DTranspose(
                    filters=filters, kernel_size=3, dilation_rate=dilation_rate, padding="same", activation="relu"
                )
            )
            self.decoder_layers.append(Dropout(rate=0.2))
        
        # Final projection layer
        self.final_layer = Conv1D(filters=1, kernel_size=1, padding="same")

    def call(self, inputs):
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = self.gru_layer(x)
        
        for layer in self.decoder_layers:
            x = layer(x)
        
        return self.final_layer(x)
    
    def fit(self, data):
        """
        Compile and train the convolutional autoencoder model on the provided data.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for training the autoencoder model.
        """
        
        self.compile(optimizer=Adam(learning_rate=0.001), loss=self.loss)

        self.fit(
            data,
            data,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0)
            ],
        )

    def predict(self, data):
        """
        Generate predictions using the trained convolutional autoencoder model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for generating predictions.

        Returns
        -------
        numpy.ndarray
            Predicted output data.
        """
        
        return super().predict(data)
