from config import *
"""
### Padding mask

When passing sequences into a transformer model, it is important that they are of uniform length. You can achieve this by padding the sequence with zeros, and truncating sentences that exceed the maximum length of your model.

In the case where the sequence is padded with zeros, we need to create a mask to let the algorithm know that it should discard the zero values.
"""

def create_padding_mask(decoder_token_ids):
    """
    Creates a matrix mask for the padding cells

    Arguments:
        decoder_token_ids -- (n, m) matrix. n=batch_size, m=fixed size of input sentences

    Returns:
        mask -- (n, 1, m) binary tensor
    """
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)

    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, :]


def create_look_ahead_mask(sequence_length):
    """
    Returns an upper triangular matrix filled with ones

    Arguments:
        sequence_length -- matrix size

    Returns:
        mask -- (size, size) tensor
    """
    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    return mask
