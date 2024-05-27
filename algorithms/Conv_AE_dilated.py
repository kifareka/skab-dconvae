from tensorflow.keras.layers import Input, Conv1D, Dropout, Conv1DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

class Conv_AE_dilated(Model): 
    """
    A reconstruction convolutional autoencoder model with dilations to detect anomalies in timeseries data using reconstruction error as an anomaly score.

    Parameters
    ----------
    dilation_rates : list of int, optional
        List of dilation rates for the convolutional layers in the encoder and decoder. Default is [1, 2, 4].
    filters : list of int, optional
        List of filter sizes for the convolutional layers in the encoder and decoder. Default is [32, 16, 8].
    loss : str, optional
        Loss function to be used for training the model. Default is "mse".

    Attributes
    ----------
    encoder_layers : list
        List of layers in the encoder part of the autoencoder.
    decoder_layers : list
        List of layers in the decoder part of the autoencoder.
    final_layer : Conv1D
        The final convolutional layer to project the output to the desired shape.

    Examples
    --------
    >>> from Conv_AE_dilated import Conv_AE_dilated
    >>> CAutoencoder = Conv_AE_dilated()
    >>> CAutoencoder.fit(train_data)
    >>> prediction = CAutoencoder.predict(test_data)
    """
    
    def __init__(
        self,
        dilation_rates=[1, 2, 4],
        filters=[32, 16, 8],
        loss="mse"
        ):
        super(Conv_AE_dilated, self).__init__()
        self._Random(0)
        self.dilation_rates = dilation_rates
        self.filters = filters
        self.loss = loss
        self.encoder_layers = []
        self.decoder_layers = []
        self._build_model()
    
    def _Random(self, seed_value): 
        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        import random
        random.seed(seed_value)

        import numpy as np
        np.random.seed(seed_value)

        import tensorflow as tf
        tf.random.set_seed(seed_value)

    def _build_model(self):
        """
        Builds a convolutional autoencoder with dilated convolutions.
        """
        # Encoder with progressive dilation rates and filters
        for dilation_rate, filters in zip(self.dilation_rates, self.filters):
            self.encoder_layers.append(
                Conv1D(
                    filters=filters, dilation_rate=dilation_rate,
                    kernel_size=3, padding="same", activation="relu"
                )
            )
            self.encoder_layers.append(Dropout(rate=0.2))
        
        # Decoder with decreasing dilation rates and filters
        for dilation_rate, filters in zip(self.dilation_rates[::-1], self.filters[::-1]):
            self.decoder_layers.append(
                Conv1DTranspose(
                    filters=filters, dilation_rate=dilation_rate,
                    kernel_size=3, padding="same", activation="relu"
                )
            )
            self.decoder_layers.append(Dropout(rate=0.2))
        
        # Final projection layer
        self.final_layer = Conv1D(filters=1, kernel_size=1, padding="same")

    def call(self, inputs):
        """
        Forward pass for the autoencoder model.
        """
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x)
        
        for layer in self.decoder_layers:
            x = layer(x)
        
        return self.final_layer(x)
    
    def fit(self, data):
        """
        Train the convolutional autoencoder model on the provided data.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for training the autoencoder model.
        """
        
        self.shape = data.shape
        self.compile(optimizer=Adam(learning_rate=0.001), loss=self.loss)

        super(Conv_AE_dilated, self).fit(
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
        
        return super(Conv_AE_dilated, self).predict(data)
