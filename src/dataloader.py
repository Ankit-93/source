from modules import *

"""## *▶ loading dataset*"""

# Data Preparation Class
class CreateDataLoaders:
    def __init__(self, data, ben_token, eng_token, ip_maxlen, out_maxlen):
        self.encoder_inps = data['bengali'].values
        self.decoder_inps = data['english'].values
        self.tknizer_eng = eng_token
        self.tknizer_ben = ben_token
        self.ip_maxlen = ip_maxlen
        self.out_maxlen = out_maxlen

    def __getitem__(self, i):
        encoder_seq = self.tknizer_ben.texts_to_sequences([self.encoder_inps[i]])
        decoder_seq = self.tknizer_eng.texts_to_sequences([self.decoder_inps[i]])
        encoder_seq = pad_sequences(encoder_seq, maxlen=self.ip_maxlen, dtype='int32', padding='post')
        decoder_seq = pad_sequences(decoder_seq, maxlen=self.out_maxlen, dtype='int32', padding='post')
        return encoder_seq[0], decoder_seq[0]


class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size=64):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.dataset.encoder_inps))
        self.on_epoch_end()
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        encoder_inputs = np.array([self.dataset[i][0] for i in batch_indexes])
        decoder_inputs = np.array([self.dataset[i][1] for i in batch_indexes])
        return encoder_inputs, decoder_inputs
    def __len__(self):
        return len(self.indexes) // self.batch_size
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)