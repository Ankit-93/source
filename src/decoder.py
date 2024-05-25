from config import *
from fnn_layer import FNNLayer
from attention import Multihead_Attention

"""# Decoder

## Decoder layer
"""

class DecoderLayer(tf.keras.layers.Layer):

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

        super(DecoderLayer, self).__init__()

        self.mha1 = Multihead_Attention(H, d_model, dk, dv)
        self.mha2 = Multihead_Attention(H, d_model, dk, dv)
        self.ffn = FNNLayer(d_model, dff)
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = LayerNormalization(epsilon=layernorm_eps)
        self.dropout_mha1 = Dropout(dropout_rate)
        self.dropout_mha2 = Dropout(dropout_rate)
        self.dropout_ffn = Dropout(dropout_rate)

    def call(self, x, encoder_output, training=False, look_ahead_mask=None, padding_mask=None):
        """
        Forward pass for the Decoder Layer

        Arguments:
            x -- Tensor of shape (batch_size, Tv, d_model)
            encoder_output --  Tensor of shape (batch_size, Tv, d_model)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers. Defaults to False
            look_ahead_mask -- Boolean mask for the target_input. Defaults to None
            padding_mask -- Boolean mask for the second multihead attention layer. Defaults to None
        Returns:
            decoder_layer_out -- Tensor of shape (batch_size, Tq, d_model)
        """
        # 1st Masked MultiHead attention
        A1 = self.mha1(x,x,x,mask=look_ahead_mask) # Self attention (batch_size, Tq, d_model)
        A1 = self.dropout_mha1(A1, training=training) #Apply Dropout during training

        #  Residual connection + Layer normalization
        out1 = self.layernorm1(x+A1)  # (batch_size, Tq, d_model)

        # 2nd Masked MultiHead attention
        A2 = self.mha2(x,encoder_output,encoder_output,mask=padding_mask) # Self attention (batch_size, Tq, d_model)
        A2 = self.dropout_mha2(A2, training=training) #Apply Dropout during training


        #  Residual connection + Layer normalization
        out2 = self.layernorm2(out1+A2)  # (batch_size, Tq, d_model)

        # Pointwise ffn
        ffn_output = self.ffn(out2) # (batch_size, Tq, d_model)
        ffn_output = self.dropout_ffn(ffn_output, training=training) # Apply Dropout during training

        # Residual connection + Layer normalization
        decoder_layer_out = self.layernorm3(ffn_output+out2)  # (batch_size, input_seq_len, fully_connected_dim)

        return decoder_layer_out



"""## Full decoder"""

class Decoder(tf.keras.layers.Layer):

    def __init__(self, N, H, d_model, dk, dv, dff, dropout_rate=0.1, layernorm_eps=1e-6):

        """
        Arguments:

        N -- number of stackeds Decoder layers (=6 in the paper)
        H -- number of heads (=8 in the paper)
        d_models -- embedding dimension (=512 in the paper)
        dk -- depth of Q and K (=64 in the paper)
        dv -- depth of V (=64 in the paper)
        dff -- the dimension of the hidden layer of the FNN (=2048 in the paper)
        dropout_rate -- Dropout parameter used (during training) before all the residual connections
        layernorm_eps -- eta regularizing parameter for the Normalization layer
        """

        super(Decoder, self).__init__()

        self.layers=[DecoderLayer(H, d_model, dk, dv, dff,
                                  dropout_rate=dropout_rate,
                                  layernorm_eps=layernorm_eps)
                                  for i in range(N)]

    def call(self, x, encoder_output, training=False, look_ahead_mask=None, padding_mask=None):
        """
        Forward pass for the Decoder Layer

        Arguments:
            x -- Tensor of shape (batch_size, Tv, d_model)
            encoder_output --  Tensor of shape (batch_size, Tv, d_model)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers. Defaults to False
            look_ahead_mask -- Boolean mask for the target_input. Defaults to None
            padding_mask -- Boolean mask for the second multihead attention layer. Defaults to None
        Returns:
            decoder_out -- Tensor of shape (batch_size, Tq, d_model)
        """

        for layer in self.layers:
            x = layer(x,encoder_output, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

        return x

