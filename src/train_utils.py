from src.modules import *
from src.encoder import *
from src.decoder import *
from src.transformer import *



class Trainer:
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
                 pe_input, pe_target, rate=0.1):

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.pe_input = pe_input
        self.pe_target = pe_target
        self.rate = rate
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.model = Transformer(
            self.num_layers, self.d_model, self.num_heads, self.dff,
            self.input_vocab_size, self.target_vocab_size,
            self.pe_input, self.pe_target, self.dropout_rate)

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def train_step(self, inp, tar, model = None):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        if not model:
            model = self.model
        with tf.GradientTape() as tape:
            predictions, _ = model(inp, tar_inp, True, None, None, None)
            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        model.save_weights('transformer_weights.h5')
        return loss, model


