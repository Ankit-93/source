from config import *
from logger import logger
from transformer import Transformer
#from data_preparation import train_dataloader, validation_dataloader



encoder_layers, attention_heads, embedding_dimension, depth_key, depth_value, dimension_fnn = 6, 8, 512, 64, 64, 2048
# dropout_rate = 0.1
# layernorm_eps=1e-6
N, H, d_model, dk, dv, dff = 6, 8, 512, 64, 64, 2048
vocab_size, T =29, 11
batch_size = 3


# transformer = Transformer(encoder_layers, attention_heads, embedding_dimension, depth_key, depth_value, dimension_fnn,
#                  vocab_size, T,
#                  dropout_rate=0.1, layernorm_eps=1e-6)

# input_shape = (None, T,vocab_size)


# x = tf.random.uniform((batch_size, T, vocab_size))
# y =  tf.random.uniform((batch_size, T, vocab_size))

# pred = transformer(x,y,training=True)
# logger.info(pred.shape)

# transformer.summary()
model = Transformer(encoder_layers, attention_heads, embedding_dimension, depth_key, depth_value, dimension_fnn,
                 vocab_size, T,
                 dropout_rate=0.1, layernorm_eps=1e-6)

# model  = Encoder_Decoder_Attention(input_vocab_size=input_vocab_size, encoder_inputs_length=22,decoder_inputs_length=25,
#                                        output_vocab_size=output_vocab_size,batch_size=1024,score_fun='general')
# model.compile(optimizer=tf.keras.optimizers.RMSprop(),loss=loss_function)
# train_steps=train.shape[0]//1024
# valid_steps=validation.shape[0]//1024
# model.fit(train_dataloader, validation_data = validation_dataloader ,steps_per_epoch=train_steps, epochs=40,callbacks=[tensor,log,checkpoint])
# model.summary()





            

