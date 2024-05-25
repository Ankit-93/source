from config import *

class FNNLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):

        """
        Arguments:
        d_model -- the dimension of the embedding (=64 in the paper)
        dff -- the dimension of the hidden layer of the FNN (=2048 in the paper)
        """

        super(FNNLayer, self).__init__()

        self.layer1 = Conv1D(filters=dff, kernel_size=1,activation="relu")
        self.layer2 = Conv1D(filters=d_model, kernel_size=1)


    def call(self, x):
        """
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len, embedding_dim)

        Returns:
            fnn_layer_out -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
        """

        x=self.layer1(x)
        fnn_layer_out=self.layer2(x)


        return fnn_layer_out

