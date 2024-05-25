from config import *
from encoder import Encoder
from decoder import Decoder
from positional_encoding import positional_encoding

"""
# Transformer

## Complete architecture
"""

class Transformer(tf.keras.Model):

    def __init__(self, N, H, d_model, dk, dv, dff,
                 vocab_size, max_positional_encoding,
                 dropout_rate=0.1, layernorm_eps=1e-6):

        super(Transformer, self).__init__()

        initializer = tf.keras.initializers.GlorotUniform()
        self.embedding = tf.Variable(initializer(shape=(vocab_size, d_model)), trainable=True)
        self.PE = positional_encoding(max_positional_encoding, d_model)

        self.dropout_encoding_input = Dropout(dropout_rate)
        self.dropout_decoding_input = Dropout(dropout_rate)

        self.encoder = Encoder(N, H, d_model, dk, dv, dff, dropout_rate=dropout_rate, layernorm_eps=layernorm_eps)
        self.decoder = Decoder(N, H, d_model, dk, dv, dff, dropout_rate=dropout_rate, layernorm_eps=layernorm_eps)



    def call(self, x, y, training=False, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None):

        x = tf.matmul(x,self.embedding)
        x = x + self.PE
        x =  self.dropout_encoding_input(x,training=training)

        encoder_output = self.encoder(x,training=training, mask=enc_padding_mask)

        y = tf.matmul(y,self.embedding)
        y = y + self.PE
        y = self.dropout_decoding_input(y,training=training)

        dec_output = self.decoder(y, encoder_output, training=training,
                                  look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask)


        pred =  tf.matmul(self.embedding,dec_output,transpose_b=True)
        pred = tf.nn.softmax(pred)

        return pred

N, H, d_model, dk, dv, dff = 6, 8, 512, 64, 64, 2048
vocab_size, T =29, 11
batch_size = 3


transformer = Transformer(N, H, d_model, dk, dv, dff,
                 vocab_size, T)

input_shape = (None, T,vocab_size)


# x = tf.random.uniform((batch_size, T, vocab_size))
# y =  tf.random.uniform((batch_size, T, vocab_size))

# pred = transformer(x,y,training=True)
# print(pred.shape)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    """ Custom loss function that will not consider the loss for padded zeros.
    why are we using this, can't we use simple sparse categorical crossentropy?
    Yes, you can use simple sparse categorical crossentropy as loss like we did in task-1. But in this loss function we are ignoring the loss
    for the padded zeros. i.e when the input is zero then we donot need to worry what the output is. This padded zeros are added from our end
    during preprocessing to make equal length for all the sentences.

    """
    
    
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

from data_preparation import *
model  = Transformer(N, H, d_model, dk, dv, dff,
                 vocab_size, T)
model.compile(optimizer=tf.keras.optimizers.RMSprop(),loss=loss_function)
train_steps=train.shape[0]//1024
valid_steps=validation.shape[0]//1024
model.fit(train_dataloader, validation_data = validation_dataloader, steps_per_epoch=train_steps, epochs=35)
model.summary()