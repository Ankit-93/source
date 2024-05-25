from config import *
from preprocessing import *
from dataloader import Dataset,Dataloder


# data = pd.read_csv('./data/ita.txt' ,sep='/n' )
with open('./data/ita.txt' , 'r',encoding='utf-8') as f:
  eng=[]
  ita=[]
  for i in f.readlines():
    ita.append(i.split('\t')[1])
    eng.append(i.split('\t')[0])

data = pd.DataFrame(data=list(zip(eng, ita)), columns=['english','italian'])
print(data.shape)
data.head()

data['english'] = data['english'].apply(preprocess)
data['italian'] = data['italian'].apply(preprocess_ita)
data.head()

df=data
df['ita_lengths'] = df['italian'].str.split().apply(len)
df = df[df['ita_lengths'] < 22 ]
df['eng_lengths'] = df['english'].str.split().apply(len)
df = df[df['eng_lengths'] < 25]
df.shape
final_data = df.drop(['eng_lengths','ita_lengths'],axis=1)
final_data=final_data
ita = final_data['italian'].values
english = final_data['english'].values
final_data.head(2)
from sklearn.model_selection import train_test_split
train,test = train_test_split(final_data,test_size=0.1, random_state=4)
train,validation = train_test_split(train,test_size=0.05, random_state=4)
print(train.shape, validation.shape,test.shape)
train['italian_inp'] ='<start> '+train['italian'].astype(str)+' <end>'
train['english_inp'] = '<start> '+train['english'].astype(str)
train['english_out'] = train['english'].astype(str)+' <end>'
validation['italian_inp'] ='<start> '+validation['italian'].astype(str)+' <end>'
validation['english_inp'] = '<start> '+validation['english'].astype(str)
validation['english_out'] = validation['english'].astype(str)+' <end>'
validation.sample(3)
train = train.drop(['english'],axis=1)
validation = validation.drop(['english'],axis=1)
train.head(2)
ita_token = Tokenizer(filters='')
ita_token.fit_on_texts(train['italian_inp'].values)

eng_token = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
eng_token.fit_on_texts(train['english_inp'].values)
eng_token.fit_on_texts(train['english_out'].values)
eng_token.word_index['<start>'],eng_token.word_index['<end>']
output_vocab_size=len(eng_token.word_index)+1
print(output_vocab_size)
input_vocab_size=len(ita_token.word_index)+1
print(input_vocab_size)
# !wget --header="Host: storage.googleapis.com" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9" --header="Referer: https://www.kaggle.com/" "https://storage.googleapis.com/kaggle-data-sets/715814/1246668/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20211204%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20211204T032627Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=937de819d58bee94ba0f85b84a0b7bdc5a065f90c89b54a1e6a3366aa756e7eee9d75837d43e3afc393a31591dc4874eff000dfbc9c43db067977724417822881292a2e40a29b711190eb26fc761fb0e3a3c6ea2235636b657d2fd9a2fe1aa4c8bbec8d45c70626fba9bae10b61886756ae9b7ce6e63850a3edb43c68dc01ac38c8e3edd2a89d78f7ef5a89b7c118d83d9a9a1ca6c8effd1c900527b5def96777dba3b136945e609020c2f37a123cb17f5ee8a42e1ae561497be67a2403417242560d7c0929e93c54d735d98666badfec5b9f2858ecae335e721d9e609dd4a1403965b4bdd51642ec7139f7d30f192847d5a27e230bfff2e6eb817e33158e586" -c -O 'archive.zip'
# ! unzip /content/archive.zip
embeddings_index = dict()
f = open('./data/glove-6B-100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix_eng = np.zeros((output_vocab_size+1, 100))
for word, i in eng_token.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix_eng[i] = embedding_vector

ita_lengths= data['italian'].str.split().apply(len)
eng_lengths=data['english'].str.split().apply(len)
ita_maxlen =np.round(np.percentile(ita_lengths,99.9))
eng_maxlen = np.round(np.percentile(eng_lengths,99.9))
train_dataset = Dataset(train, ita_token, eng_token, int(ita_maxlen),int(eng_maxlen))
validation_dataset  = Dataset(validation, ita_token, eng_token, int(ita_maxlen),int(eng_maxlen))

train_dataloader = Dataloder(train_dataset, batch_size=1024)
validation_dataloader = Dataloder(validation_dataset, batch_size=1024)

print(train_dataloader[0][0][0].shape, train_dataloader[0][0][1].shape, train_dataloader[0][1].shape)
