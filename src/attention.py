from config import *


"""# Sub layers

## Attention

### Single head attention
"""

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Calculate the attention weights.

    Arguments:
        Q -- query shape == (..., Tq, dk)
        K -- key shape == (..., Tv, dk)
        V -- value shape == (..., Tv, dv)
        mask: Float tensor with shape broadcastable to (..., Tq, Tv). Defaults to None.

    Returns:
        output -- (attention,attention_weights)
    """

    #Compute the scaled dot-product Q•K
    matmul_QK = tf.matmul(Q,K,transpose_b=True)  # dot-product of shape (..., Tq, Tv)

    dk = K.shape[-1]
    scaled_attention_logits = matmul_QK/np.sqrt(dk) # scaled dot-product of shape (..., Tq, Tv)

    # Add the mask to the scaled dot-product
    if mask is not None:
        scaled_attention_logits += (1. - mask) *(-1e9)

    # Compute the Softmax
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # weights of shape (..., Tq, Tv)

    #Multiply with V
    output = tf.matmul(attention_weights,V)  # Attention representation of shape (..., Tq, dv)

    return output, attention_weights



"""### MultiHead Attention"""

class Multihead_Attention(tf.keras.layers.Layer):
    def __init__(self, H, d_model, dk, dv):

        """
        Arguments:
        H -- number of heads (=8 in the paper)
        d_models -- embedding dimension (=512 in the paper)
        dk -- depth of Q and K (=64 in the paper)
        dv -- depth of V (=64 in the paper)
        """

        super(Multihead_Attention, self).__init__()

        initializer = tf.keras.initializers.GlorotUniform()
        self.WQ = tf.Variable(initializer(shape=(H, d_model, dk)), trainable=True)
        self.WK = tf.Variable(initializer(shape=(H, d_model, dk)), trainable=True)
        self.WV = tf.Variable(initializer(shape=(H, d_model, dv)), trainable=True)
        self.WO = tf.Variable(initializer(shape=(H*dv,d_model)), trainable=True)


    def call(self, Q, K, V, mask=None):
        """
        Calculate the attention weights.

        Arguments:
            Q -- query shape == (..., Tq, d_model)
            K -- key shape == (..., Tv, d_model)
            V -- value shape == (..., Tv, d_model)
            mask: Float tensor with shape broadcastable to (..., Tq, Tv). Defaults to None.

        Returns:
            output -- Multihead attention A of shape (batch_size, Tq, d_model)
        """
        #Projecting Q,K,V to Qh, Kh, Vh. The H projection are stacked on the penultiem axis
        Qh= tf.experimental.numpy.dot(Q, self.WQ) #of shape (batch_size, Tq, H, dk)
        Kh= tf.experimental.numpy.dot(K, self.WK) #of shape (batch_size, Tv, H, dk)
        Vh= tf.experimental.numpy.dot(V, self.WV) #of shape (batch_size, Tv, H, dv)

        #Transposition
        Qh=tf.transpose(Qh, [0,2,1,3]) #of shape (batch_size, H, Tq, dk)
        Kh=tf.transpose(Kh, [0,2,1,3]) #of shape (batch_size, H, Tv, dk)
        Vh=tf.transpose(Vh, [0,2,1,3]) #of shape (batch_size, H, Tv, dv)

        # Computing the dot-product attention
        Ah,_=scaled_dot_product_attention(Qh, Kh, Vh, mask=mask) #of shape (batch_size, H, Tq, dv)

        """
        Flattening the H and dv axis and projecting back to d_model
        A = tf.reshape(Ah,(*Ah.shape[:-2],-1))"""
        s=Ah.shape
        A = tf.reshape(Ah,(s[0],s[2],s[1]*s[3])) #of shape (batch_size, Tq, H*dv)
        A= tf.experimental.numpy.dot(A, self.WO) #of shape (batch_size, Tq, d_model)

        return A

## DEBUG

H, d_model, dk, dv=8,512,64,32
batch_size, Tq, Tv = 16,9,9

mha_layer= Multihead_Attention(H, d_model, dk, dv)

Q= tf.random.uniform((batch_size, Tq, d_model))
K= tf.random.uniform((batch_size, Tv, d_model))
V= tf.random.uniform((batch_size, Tv, d_model))

A=mha_layer(Q,K,V)
print(A.shape)