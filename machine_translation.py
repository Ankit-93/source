# -*- coding: utf-8 -*-
"""Machine Translation.ipynb"""

import os
import pickle
import numpy as np
import tensorflow as tf
from source.logger import *

# Load pre-processed data
with open('./source/data/translationPICKLE', 'rb') as fp:
    PICK = pickle.load(fp)

vocab_eng, vocab_beng = PICK[0], PICK[1]
vocab_len = len(vocab_beng)
np_embedding_eng, np_embedding_beng = map(lambda x: np.asarray(x, np.float32), PICK[2:4])
word_vec_dim = np_embedding_eng.shape[1]

# Debug: print shapes and lengths of the loaded data
logger.info(f'vocab_eng: {len(vocab_eng)}, vocab_beng: {len(vocab_beng)}')
logger.info(f'np_embedding_eng shape: {np_embedding_eng.shape}, np_embedding_beng shape: {np_embedding_beng.shape}')

# Verify if batches exist and print their shapes
train_batch_x, train_batch_y, val_batch_x, val_batch_y, test_batch_x, test_batch_y = PICK[4:]
logger.info(f'train_batch_x length: {len(train_batch_x)}, train_batch_y length: {len(train_batch_y)}')
logger.info(f'val_batch_x length: {len(val_batch_x)}, val_batch_y length: {len(val_batch_y)}')
logger.info(f'test_batch_x length: {len(test_batch_x)}, test_batch_y length: {len(test_batch_y)}')

if not (train_batch_x and train_batch_y and test_batch_x and test_batch_y):
    raise ValueError("One of the batch lists is empty. Please check the data loading process.")

# Extract the first batch for training, validation, and testing
train_x, train_y = train_batch_x[0], train_batch_y[0]
test_x, test_y = test_batch_x[0], test_batch_y[0]

logger.info(f'train_x shape: {train_x.shape}, train_y shape: {train_y.shape}')
logger.info(f'test_x shape: {test_x.shape}, test_y shape: {test_y.shape}')

def most_similar_eucli_eng(x):
    dists = np.linalg.norm(np_embedding_eng - x, axis=1)
    return np.argsort(dists)

def vec2word_eng(vec):
    return vocab_eng[most_similar_eucli_eng(np.asarray(vec, np.float32))[0]]

# Hyperparameters and Placeholders
h, N, dqkv, d = 8, 1, 32, 1024
learning_rate, epochs, keep_prob = 0.001, 200, 0.5

x = tf.keras.Input(shape=(None, word_vec_dim), name='input_x')
y = tf.keras.Input(shape=(None,), name='input_y')
output_len = tf.keras.Input(shape=(), dtype=tf.int32, name='output_len')
teacher_forcing = tf.keras.Input(shape=(), dtype=tf.bool, name='teacher_forcing')
tf_pad_mask = tf.keras.Input(shape=(None, None), name='pad_mask')
tf_illegal_position_masks = tf.keras.Input(shape=(None, None, None), name='illegal_position_masks')
tf_pe_out = tf.keras.Input(shape=(None, None), name='positional_encoding')

# Model Parameters
Wq_enc = tf.Variable(tf.random.truncated_normal([N, h, word_vec_dim, dqkv], stddev=0.01))
Wk_enc = tf.Variable(tf.random.truncated_normal([N, h, word_vec_dim, dqkv], stddev=0.01))
Wv_enc = tf.Variable(tf.random.truncated_normal([N, h, word_vec_dim, dqkv], stddev=0.01))
Wo_enc = tf.Variable(tf.random.truncated_normal([N, h * dqkv, word_vec_dim], stddev=0.01))

W1_enc = tf.Variable(tf.random.truncated_normal([N, 1, 1, word_vec_dim, d], stddev=0.01))
b1_enc = tf.Variable(tf.zeros([N, d]))
W2_enc = tf.Variable(tf.random.truncated_normal([N, 1, 1, d, word_vec_dim], stddev=0.01))
b2_enc = tf.Variable(tf.zeros([N, word_vec_dim]))

Wq_dec_1 = tf.Variable(tf.random.truncated_normal([N, h, word_vec_dim, dqkv], stddev=0.01))
Wk_dec_1 = tf.Variable(tf.random.truncated_normal([N, h, word_vec_dim, dqkv], stddev=0.01))
Wv_dec_1 = tf.Variable(tf.random.truncated_normal([N, h, word_vec_dim, dqkv], stddev=0.01))
Wo_dec_1 = tf.Variable(tf.random.truncated_normal([N, h * dqkv, word_vec_dim], stddev=0.01))
Wq_dec_2 = tf.Variable(tf.random.truncated_normal([N, h, word_vec_dim, dqkv], stddev=0.01))
Wk_dec_2 = tf.Variable(tf.random.truncated_normal([N, h, word_vec_dim, dqkv], stddev=0.01))
Wv_dec_2 = tf.Variable(tf.random.truncated_normal([N, h, word_vec_dim, dqkv], stddev=0.01))
Wo_dec_2 = tf.Variable(tf.random.truncated_normal([N, h * dqkv, word_vec_dim], stddev=0.01))

W1_dec = tf.Variable(tf.random.truncated_normal([N, 1, 1, word_vec_dim, d], stddev=0.01))
b1_dec = tf.Variable(tf.zeros([N, d]))
W2_dec = tf.Variable(tf.random.truncated_normal([N, 1, 1, d, word_vec_dim], stddev=0.01))
b2_dec = tf.Variable(tf.zeros([N, word_vec_dim]))

scale_enc_1, shift_enc_1 = tf.Variable(tf.ones([N, 1, 1, word_vec_dim])), tf.Variable(tf.zeros([N, 1, 1, word_vec_dim]))
scale_enc_2, shift_enc_2 = tf.Variable(tf.ones([N, 1, 1, word_vec_dim])), tf.Variable(tf.zeros([N, 1, 1, word_vec_dim]))
scale_dec_1, shift_dec_1 = tf.Variable(tf.ones([N, 1, 1, word_vec_dim])), tf.Variable(tf.zeros([N, 1, 1, word_vec_dim]))
scale_dec_2, shift_dec_2 = tf.Variable(tf.ones([N, 1, 1, word_vec_dim])), tf.Variable(tf.zeros([N, 1, 1, word_vec_dim]))
scale_dec_3, shift_dec_3 = tf.Variable(tf.ones([N, 1, 1, word_vec_dim])), tf.Variable(tf.zeros([N, 1, 1, word_vec_dim]))
def positional_encoding(seq_len, model_dimensions):
    pe = np.zeros((seq_len, model_dimensions), np.float32)
    for pos in range(seq_len):
        for i in range(model_dimensions):
            pe[pos, i] = np.sin(pos / (10000 ** (2 * i / model_dimensions)))
    return pe.reshape((seq_len, model_dimensions))

def layer_norm(inputs, scale, shift, epsilon=1e-5):
    mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
    return scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift

def generate_masks_for_illegal_positions(out_len):
    masks = np.full((out_len-1, out_len, out_len), -2**30, dtype=np.float32)
    for i in range(1, out_len):
        masks[i-1, i:, :] = -2**30
        masks[i-1, :, i:] = -2**30
    return masks

def attention(Q, K, V, d, filled=0, mask=False):
    K = tf.transpose(K, [1, 0])

    #K = tf.transpose(K, [0, 2, 1])
    d = tf.cast(d, tf.float32)
    softmax_component = tf.matmul(Q, K) / tf.sqrt(d)
    if mask:
        softmax_component += tf_illegal_position_masks[filled-1]
    return tf.matmul(tf.nn.dropout(tf.nn.softmax(softmax_component), keep_prob), V)

def multihead_attention(Q, K, V, d, weights, filled=0, mask=False):
    Q_, K_, V_ = map(lambda x: tf.reshape(x, [-1, tf.shape(Q)[2]]), (Q, K, V))
    heads = []
    for i in range(h):
        Q_w = tf.matmul(Q_, weights['Wq'][i])
        K_w = tf.matmul(K_, weights['Wk'][i])
        V_w = tf.matmul(V_, weights['Wv'][i])
        heads.append(attention(Q_w, K_w, V_w, d, filled, mask))
    combined_heads = tf.concat(heads, axis=2)
    output = tf.matmul(tf.reshape(combined_heads, [-1, h * d]), weights['Wo'])
    return tf.reshape(output, tf.shape(Q))

def encoder(x, weights, attention_weights, dqkv):
    W1, W2, b1, b2 = weights['W1'], weights['W2'], weights['b1'], weights['b2']
    scale1, shift1, scale2, shift2 = weights['scale1'], weights['shift1'], weights['scale2'], weights['shift2']
    sublayer1 = multihead_attention(x, x, x, dqkv, attention_weights)
    sublayer1 = layer_norm(tf.nn.dropout(sublayer1, keep_prob) + x, scale1, shift1)
    sublayer1_ = tf.reshape(sublayer1, [-1, 1, tf.shape(sublayer1)[1], word_vec_dim])
    sublayer2 = tf.nn.conv2d(sublayer1_, W1, strides=[1, 1, 1, 1], padding='SAME')
    sublayer2 = tf.nn.relu(tf.nn.bias_add(sublayer2, b1))
    sublayer2 = tf.nn.conv2d(sublayer2, W2, strides=[1, 1, 1, 1], padding='SAME')
    sublayer2 = layer_norm(tf.nn.dropout(tf.reshape(tf.nn.bias_add(sublayer2, b2), [-1, tf.shape(sublayer2)[2], word_vec_dim]) + sublayer1, keep_prob), scale2, shift2)
    return sublayer2

def decoder_block(x, en_out, weights, attention_weights, filled, dqkv):
    W1, W2, b1, b2 = weights['W1'], weights['W2'], weights['b1'], weights['b2']
    scale1, shift1, scale2, shift2, scale3, shift3 = weights['scale1'], weights['shift1'], weights['scale2'], weights['shift2'], weights['scale3'], weights['shift3']
    sublayer1 = multihead_attention(x, x, x, dqkv, attention_weights[0], filled, True)
    sublayer1 = layer_norm(tf.nn.dropout(sublayer1, keep_prob) + x, scale1, shift1)
    sublayer2 = multihead_attention(sublayer1, en_out, en_out, dqkv, attention_weights[1])
    sublayer2 = layer_norm(tf.nn.dropout(sublayer2, keep_prob) + sublayer1, scale2, shift2)
    sublayer2_ = tf.reshape(sublayer2, [-1, 1, tf.shape(sublayer2)[1], word_vec_dim])
    sublayer3 = tf.nn.conv2d(sublayer2_, W1, strides=[1, 1, 1, 1], padding='SAME')
    sublayer3 = tf.nn.relu(tf.nn.bias_add(sublayer3, b1))
    sublayer3 = tf.nn.conv2d(sublayer3, W2, strides=[1, 1, 1, 1], padding='SAME')
    sublayer3 = layer_norm(tf.nn.dropout(tf.reshape(tf.nn.bias_add(sublayer3, b2), [-1, tf.shape(sublayer3)[2], word_vec_dim]) + sublayer2), scale3, shift3)
    return sublayer3

def encoder_stack(x):
    for i in range(N):
        x = encoder(x, {
            'W1': W1_enc[i], 'W2': W2_enc[i], 'b1': b1_enc[i], 'b2': b2_enc[i],
            'scale1': scale_enc_1[i], 'shift1': shift_enc_1[i], 'scale2': scale_enc_2[i], 'shift2': shift_enc_2[i]
        }, {'Wq': Wq_enc[i], 'Wk': Wk_enc[i], 'Wv': Wv_enc[i], 'Wo': Wo_enc[i]}, dqkv)
    return x

def decoder_stack2(x, en_out):
    for i in range(N):
        x = decoder_block(x, en_out, {
            'W1': W1_dec[i], 'W2': W2_dec[i], 'b1': b1_dec[i], 'b2': b2_dec[i],
            'scale1': scale_dec_1[i], 'shift1': shift_dec_1[i], 'scale2': scale_dec_2[i], 'shift2': shift_dec_2[i],
            'scale3': scale_dec_3[i], 'shift3': shift_dec_3[i]
        }, [{'Wq': Wq_dec_1[i], 'Wk': Wk_dec_1[i], 'Wv': Wv_dec_1[i], 'Wo': Wo_dec_1[i]}, {'Wq': Wq_dec_2[i], 'Wk': Wk_dec_2[i], 'Wv': Wv_dec_2[i], 'Wo': Wo_dec_2[i]}], tf.shape(x)[1], dqkv)
    return x

def decoder_stack(x, en_out):
    for i in range(N):
        x = decoder_block(x, en_out, {
            'W1': W1_dec[i], 'W2': W2_dec[i], 'b1': b1_dec[i], 'b2': b2_dec[i],
            'scale1': scale_dec_1[i], 'shift1': shift_dec_1[i], 'scale2': scale_dec_2[i], 'shift2': shift_dec_2[i],
            'scale3': scale_dec_3[i], 'shift3': shift_dec_3[i]
        }, [{'Wq': Wq_dec_1[i], 'Wk': Wk_dec_1[i], 'Wv': Wv_dec_1[i], 'Wo': Wo_dec_1[i]}, {'Wq': Wq_dec_2[i], 'Wk': Wk_dec_2[i], 'Wv': Wv_dec_2[i], 'Wo': Wo_dec_2[i]}], tf.shape(x)[1], dqkv)
        # Check the data types of x and tf_pe_out[:,:tf.shape(x)[1],:]
        if x.dtype != tf_pe_out[:,:tf.shape(x)[1],:].dtype:
            x = tf.cast(x, tf_pe_out[:,:tf.shape(x)[1],:].dtype)
        # Add x and tf_pe_out[:,:tf.shape(x)[1],:]
        x += tf_pe_out[:,:tf.shape(x)[1],:]
    return x

tf_pe = positional_encoding(500, word_vec_dim)
pe = tf.constant(tf_pe.reshape(1, 500, word_vec_dim), dtype=tf.float32)

def transformer(x, y):
    enc_out = encoder_stack(x + tf_pe_out[:,:tf.shape(x)[1],:])
    dec_in = tf.pad(y[:, :-1], [[0, 0], [1, 0]])
    if teacher_forcing is not None:
        for i in range(1, tf.shape(y)[1]):
            dec_in = tf.concat([dec_in[:, :i], y[:, i:]], axis=1)
            print("Decoder",dec_in.shape)
            dec_out = decoder_stack(tf.cast(dec_in, tf.float32) + tf_pe_out[:,:tf.shape(dec_in)[1],:], enc_out)
            logits = tf.matmul(dec_out, np_embedding_beng.T)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    else:
        dec_out = decoder_stack(dec_in + tf_pe_out[:,:tf.shape(dec_in)[1],:], enc_out)
        logits = tf.matmul(dec_out, np_embedding_beng.T)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return loss, train_op

# Training loop
for epoch in range(epochs):
    train_loss, train_op = transformer(train_x, train_y)
    val_loss, _ = transformer(test_x, test_y)
    if epoch % 10 == 0:
        logger.info(f'Epoch {epoch}, Train Loss: {train_loss.numpy()}, Val Loss: {val_loss.numpy()}')