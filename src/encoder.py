from config import *
from fnn_layer import FNNLayer
from attention import Multihead_Attention
"""# Encoder

## Encoder layer

We recall that the encoder layer is composed by a multi-head self-attention mechanism, followed by a positionwise fully connected feed-forward network. This archirecture includes a residual connection around each of the two sub-layers, followed by layer normalization.
"""

class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, H, d_model, dk, dv, dff, dropout_rate=0.1, layernorm_eps=1e-6):

        """
        Arguments:

        H -- number of heads (=8 in the paper)
        d_models -- embedding dimension (=512 in the paper)
        dk -- depth of Q and K (=64 in the paper)
        dv -- depth of V (=64 in the paper)
        dff -- the dimension of the hidden layer of the FNN (=2048 in the paper)
        dropout_rate -- Dropout parameter used (during training) before all the residual connections
        layernorm_eps -- eta regularizing parameter for the Normalization layer
        """

        super(EncoderLayer, self).__init__()

        self.mha = Multihead_Attention(H, d_model, dk, dv)
        self.ffn = FNNLayer(d_model, dff)
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.dropout_mha = Dropout(dropout_rate)
        self.dropout_ffn = Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        """
        Forward pass for the Encoder Layer

        Arguments:
            x -- Tensor of shape (batch_size, Tq, d_model)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers. Defaults to False
            mask -- Boolean mask to ensure that the padding is not
                    treated as part of the input. Defaults to None
        Returns:
            encoder_layer_out -- Tensor of shape (batch_size, Tq, d_model)
        """
        A = self.mha(x,x,x,mask=mask) # Self attention (batch_size, Tq, d_model)
        A = self.dropout_mha(A, training=training) #Apply Dropout during training


        #  Residual connection + Layer normalization
        out1 = self.layernorm1(x+A)  # (batch_size, Tq, d_model)

        # Pointwise ffn
        ffn_output = self.ffn(out1) # (batch_size, Tq, d_model)
        ffn_output = self.dropout_ffn(ffn_output, training=training) # Apply Dropout during training

        # Residual connection + Layer normalization
        encoder_layer_out = self.layernorm2(ffn_output+out1)  # (batch_size, input_seq_len, fully_connected_dim)

        return encoder_layer_out



"""## Full Encoder"""

class Encoder(tf.keras.layers.Layer):

    def __init__(self, N, H, d_model, dk, dv, dff, dropout_rate=0.1, layernorm_eps=1e-6):
        """
        Arguments:

        N -- number of stackeds Encoder layers (=6 in the paper)
        H -- number of heads (=8 in the paper)
        d_models -- embedding dimension (=512 in the paper)
        dk -- depth of Q and K (=64 in the paper)
        dv -- depth of V (=64 in the paper)
        dff -- the dimension of the hidden layer of the FNN (=2048 in the paper)
        dropout_rate -- Dropout parameter used (during training) before all the residual connections
        layernorm_eps -- eta regularizing parameter for the Normalization layer
        """

        super(Encoder, self).__init__()

        self.layers=[EncoderLayer(H, d_model, dk, dv, dff,
                                  dropout_rate=dropout_rate,
                                  layernorm_eps=layernorm_eps)
                                  for i in range(N)]

    def call(self, x, training=False, mask=None):
        """
        Forward pass for the Encoder

        Arguments:
            x -- Tensor of shape (batch_size, Tq, d_model)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers. Defaults to False
            mask -- Boolean mask to ensure that the padding is not
                    treated as part of the input. Defaults to None
        Returns:
            encoder_out -- Tensor of shape (batch_size, Tq, d_model)
        """

        for layer in self.layers:
            x = layer(x, training=training, mask=mask)

        return x

