from app import *


# Load the model weights
model.load_weights('path_to_save_weights/transformer_weights.h5')

# Translation function
def preprocess_sentence(sentence, tokenizer, max_length):
    sentence = preprocess_sentence(sentence)
    sentence_seq = tokenizer.texts_to_sequences([sentence])
    sentence_seq = pad_sequences(sentence_seq, maxlen=max_length, padding='post')
    return sentence_seq

def translate(sentence, inp_lang, targ_lang, max_length_inp, max_length_targ):
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    enc_padding_mask = create_padding_mask(inputs)
    enc_output = transformer.encoder(inputs, False, enc_padding_mask)
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    for t in range(max_length_targ):
        look_ahead_mask = create_look_ahead_mask(tf.shape(dec_input)[1])
        dec_padding_mask = create_padding_mask(inputs)
        predictions, attention_weights = transformer.decoder(dec_input, enc_output, False, look_ahead_mask, dec_padding_mask)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if targ_lang.index_word[predicted_id.numpy()[0][0]] == '<end>':
            return result
        result += targ_lang.index_word[predicted_id.numpy()[0][0]] + ' '
        dec_input = tf.concat([dec_input, predicted_id], axis=-1)
    return result

# Example usage
input_sentence = "আপনি কেমন আছেন?"
translation = translate(input_sentence, inp_lang, targ_lang, max_length_inp, max_length_targ)
print(f'Translation: {translation}')
