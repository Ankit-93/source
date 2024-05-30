from modules import *



"""## *▶ loading dataset*"""
class Training_Utils:

    def __init__(self, data, ben_token, eng_token, ip_maxlen, out_maxlen):
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(self,real, pred, loss_object):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


    def train_step(self,inp, tar , model):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        with tf.GradientTape() as tape:
            predictions, _ = model(inp, tar_inp, True, None, None, None)
            loss = self.loss_function(tar_real, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss


    def evaluate_(self,inp_sentence, tar_sentence, tar_tokenizer, model):
        tar_inp = tar_sentence[:, :-1]
        tar_real = tar_sentence[:, 1:]
        predictions, _ = model(inp_sentence, tar_inp, False, None, None, None)
        translated_sentence = [[tar_tokenizer.index_word[np.argmax(word)] for word in sentence] 
                                    for sentence in predictions ]
        for sentence in translated_sentence:
            print(sentence)
        loss = self.loss_function(tar_real, predictions)
        return loss

    # Evaluation loop
    def evaluate_model(self,val_dataloader):
        total_loss = 0
        for batch, (inp, tar) in enumerate(val_dataloader):
            batch_loss = self.evaluate_(inp, tar)
            total_loss += batch_loss
        return total_loss.numpy() / len(val_dataloader)

