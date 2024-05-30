
from src.modules import *

"""## ▶ loading dataset"""

from src.logger import logger
from src.data_processing import *
from src.train_utils import Trainer
from src.transformer import Transformer




input_tensor, target_tensor, inp_lang, targ_lang = load_dataset("./source/data/ben.txt")
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
logger.info(" Max_length of the input {} & target tensors {}".format(max_length_targ, max_length_inp))
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
logger.info("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
logger.info("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[0])

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).shuffle(BUFFER_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)


num_layers = 2
d_model = 512
num_heads = 2
dff = 2048
input_vocab_size = len(inp_lang.word_index)+1
target_vocab_size = len(targ_lang.word_index)+1
pe_input = 10000
pe_target = 6000
dropout_rate = 0.05
batch_size = 64
# model = Transformer(
#     num_layers, d_model, num_heads, dff,
#     input_vocab_size, target_vocab_size,
#     pe_input, pe_target, dropout_rate)

trainer = Trainer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
                 pe_input, pe_target, rate=0.1)
EPOCHS = 2

for epoch in range(EPOCHS):
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
        if model :
            batch_loss, model = trainer.train_step(inp, targ)
        else:
            batch_loss, model = trainer.train_step(inp, targ, model)
        total_loss += batch_loss

    print(f'Epoch {epoch + 1}, Loss: {total_loss.numpy() / batch_size}')

# Model summary
model.summary()



# def evaluate_(inp_sentence, tar_sentence):
#     tar_inp = tar_sentence[:, :-1]
#     tar_real = tar_sentence[:, 1:]

#     predictions, _ = model(inp_sentence, tar_inp, False, None, None, None)
#     translated_sentence = [[inp_lang.index_word[np.argmax(word)] for word in sentence] for sentence in predictions ]
#     for sentence in translated_sentence:
#         print(sentence)
#     loss = loss_function(tar_real, predictions)

#     return loss

# # Evaluation loop
# def evaluate_model(val_dataloader):
#     total_loss = 0

#     for batch, (inp, tar) in enumerate(val_dataloader):
#         batch_loss = evaluate_(inp, tar)
#         total_loss += batch_loss

#     return total_loss.numpy() / len(val_dataloader)

# validation_loss = evaluate_model(val_dataset)
# print(f'Validation Loss: {validation_loss}')