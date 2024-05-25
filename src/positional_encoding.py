from config import *
"""## Positional encoding"""

def positional_encoding(positions, d):
    """
    Precomputes a matrix with all the positional encodings

    Arguments:
        positions (int) -- Maximum number of positions to be encoded
        d (int) -- Encoding size d_model

        arguments de get_angles:
            pos -- Column vector containing the positions [[0], [1], ...,[N-1]]
            k --   Row vector containing the dimension span [[0, 1, 2, ..., d-1]]
            d(integer) -- Encoding size

    Returns:
        pos_encoding -- (1, position, d_model) A matrix with the positional encodings
    """
    # initialize a matrix angle_rads of all the angles
    pos=np.arange(positions)[:, np.newaxis] #Column vector containing the position span [0,1,..., positions]
    k= np.arange(d)[np.newaxis, :]  #Row vector containing the dimension span [[0, 1, ..., d-1]]
    i = k//2
    angle_rads = pos/(10000**(2*i/d)) #Matrix of angles indexed by (pos,i)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    #adds batch axis
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

"""Let us plot the obtained results"""

positions, d=50,512
pos_encoding = positional_encoding(positions, d)

print (pos_encoding.shape)

plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('d')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()