"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                                This function is to Debug
"""


from logger import logger
logger.info("""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

                                              This function is to Debug
""")


from config import *
from attention import scaled_dot_product_attention
logger.info("""----------- Single head attention -----------""")

batch_size,Tq, Tv, dk, dv= 16,9,9,64,128 # we need Tq=Tv
Q = tf.random.uniform((batch_size,Tq, dk))
K = tf.random.uniform((batch_size,Tv, dk))
V = tf.random.uniform((batch_size,Tv, dv))
logger.info(f"Shape of Query: {Q.shape}, Key: {K.shape} & Value: {V.shape}")
A,_= scaled_dot_product_attention(Q, K, V)
logger.info(f"Shape of Attention: {A.shape}")

from attention import Multihead_Attention
logger.info("----------- Multi head attention -----------")
H, d_model, dk, dv=8,512,64,32
batch_size, Tq, Tv = 16,9,9
mha_layer= Multihead_Attention(H, d_model, dk, dv)
Q = tf.random.uniform((batch_size, Tq, d_model))
K = tf.random.uniform((batch_size, Tv, d_model))
V = tf.random.uniform((batch_size, Tv, d_model))
logger.info(f"Number of heads:{H}, embedding dimension:{d_model}, depth of Query:{dk}, Key:{dk} and Value:{dv}")
A = mha_layer(Q,K,V)
logger.info(f"Shape of Multi Head Attention: {A.shape}")

logger.info("----------- Pointwise FNN -----------")
from fnn_layer import FNNLayer
d_model, dff = 64, 2048
logger.info(f"Embedding Dimension:{d_model},Hidden Layer Dimension:{dff}")
fnn_layer= FNNLayer(d_model, dff)

batch_size, Tv= 16, 9
x=tf.random.uniform((batch_size, Tv, d_model))

logger.info(f"With Batch Size:{batch_size} and Tv :{Tv}; FNN_Layer output:{fnn_layer(x).shape}")

logger.info("----------- Positional Encoding -----------")

from positional_encoding import positional_encoding
positions, d=50,512
pos_encoding = positional_encoding(positions, d)

logger.info(f"Positional Encoding {pos_encoding.shape} with {positions} Positions & Depth:{d}")

# plt.pcolormesh(pos_encoding[0], cmap='RdBu')
# plt.xlabel('d')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()

logger.info("----------- Padding mask -----------")
from mask import create_padding_mask, create_look_ahead_mask

x = tf.constant([[7., 6., 0., 0., 1.], [1., 2., 3., 0., 0.], [0., 0., 0., 4., 5.]])
logger.info(create_padding_mask(x))

x = tf.random.uniform((1, 3))
logger.info(create_look_ahead_mask(x.shape[1]))

logger.info("----------- Encoder -----------")
from encoder import EncoderLayer,Encoder
## DEBUG

H, d_model, dk, dv, dff = 8, 512, 64, 32, 2048
layer = EncoderLayer(H, d_model, dk, dv, dff)

batch_size, Tq= 43, 27
x = tf.random.uniform((batch_size, Tq, d_model))

output = layer(x,training=True)
logger.info(output.shape)

## DEBUG

N,H, d_model, dk, dv, dff = 6,8, 512, 64, 32, 2048
encoder = Encoder(N,H, d_model, dk, dv, dff)

batch_size, Tq= 43, 27
x = tf.random.uniform((batch_size, Tq, d_model))

output = encoder(x,training=True)
logger.info(output.shape)

logger.info("----------- Decoder -----------")

from decoder import DecoderLayer, Decoder

H, d_model, dk, dv, dff = 8, 512, 64, 32, 2048
layer = DecoderLayer(H, d_model, dk, dv, dff)

batch_size, Tq, Tv= 43, 57, 57
x = tf.random.uniform((batch_size, Tv, d_model))
encoder_output = tf.random.uniform((batch_size, Tq, d_model))

output = layer(x,encoder_output,training=True)
logger.info(output.shape)

## DEBUG

N,H, d_model, dk, dv, dff = 6,8, 512, 64, 32, 2048
decoder = Decoder(N,H, d_model, dk, dv, dff)

batch_size, Tq, Tv= 43, 57, 57
x = tf.random.uniform((batch_size, Tv, d_model))
encoder_output = tf.random.uniform((batch_size, Tq, d_model))

output = decoder(x,encoder_output,training=True)
logger.info(output.shape)

logger.info("""----------- TransFormer -----------""")
from transformer import Transformer
N, H, d_model, dk, dv, dff = 6, 8, 512, 64, 64, 2048
vocab_size, T =29, 11
batch_size = 3


transformer = Transformer(N, H, d_model, dk, dv, dff,
                 vocab_size, T)

input_shape = (None, T,vocab_size)


x = tf.random.uniform((batch_size, T, vocab_size))
y =  tf.random.uniform((batch_size, T, vocab_size))

pred = transformer(x,y,training=True)
logger.info(pred.shape)

transformer.summary()


