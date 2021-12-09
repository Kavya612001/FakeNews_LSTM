import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
import gensim
from tensorflow.keras.models import load_model
#import preprocess_kgptalkie as ps


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,LSTM,Conv1D,MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score


def get_weight_matrix(model):
    DIM=100
    tokenizer=Tokenizer()
    vocab_size=len(tokenizer.word_index)+1
    vocab = tokenizer.word_index
    weight_matrix = np.zeros((vocab_size,DIM))
    for word, i in vocab.items():
        weight_matrix[i] = model.wv[word]
    return weight_matrix

def model1():
    fake=pd.read_csv("Fake.csv")
    print(fake.head(2))
    real=pd.read_csv("True.csv")
    unknown_publishers = []
    for index , row in enumerate(real.text.values):
        try:
            record=row.split('-',maxsplit=1)
            record[1]

            assert(len(record[0])<120)
        except:
            unknown_publishers.append(index)

    real=real.drop(8970,axis=0)

    publisher=[]
    tmp_text=[]
    for index , row in enumerate(real.text.values):
        if index in unknown_publishers:
            tmp_text.append(row)
            publisher.append("unknown")
        else:
            record=row.split('-',maxsplit=1)
            publisher.append(record[0].strip())
            tmp_text.append(record[1].strip())

    real['publisher']=publisher
    real['text']=tmp_text
    print(real.head())
    empty_fake_index = [index for index,text in enumerate(fake.text.tolist()) if str(text).strip()==""]
    real['text']=real['title']+" "+real['text']

    fake['text']=fake['title']+" "+fake['text']

    real['text']=real['text'].apply(lambda x : str(x).lower())
    fake['text']=fake['text'].apply(lambda x : str(x).lower())

    real['class']=1
    fake['class']=0
    real=real[['text','class']]

    fake=fake[['text','class']]

    data = real.append(fake,ignore_index=True)
    print(data.sample(5))

   #import gensim

    y = data['class'].values

    X = [d.split() for d in data['text'].tolist()]
    
    DIM=100
    w2v_model=gensim.models.Word2Vec(sentences=X,vector_size=100,window=10,min_count=1)
    len(w2v_model.wv.vocab)
    
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(X)

    X=tokenizer.texts_to_sequences(X)
    nos = np.array([len(x) for x in X])
    len(nos[nos>1000])
    maxlen=1000
    X=pad_sequences(X,maxlen=maxlen)
    vocab_size=len(tokenizer.word_index)+1
    vocab = tokenizer.word_index
    embedding_vectors = get_weight_matrix(w2v_model)
    model=Sequential()
    model.add(Embedding(vocab_size,output_dim=DIM,weights=[embedding_vectors],input_length=maxlen,trainable=False))
    model.add(LSTM(units=128))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

    model.summary()

    X_train,X_test,y_train,y_test=train_test_split(X,y)

    model.fit(X_train,y_train,validation_split=0.3,epochs=6)

    y_pred=(model.predict(X_test)>=0.5).astype(int)

    print(accuracy_score(y_test,y_pred))

    print(classification_report(y_test,y_pred))



def test(newstofind):
    x=[]
    
    tokenizer=Tokenizer()
    #model1()
    new_model = load_model('lstmmodel.h5')
    x.append(newstofind)
    print(x)    
#x=['']
    maxlen=1000
    x=tokenizer.texts_to_sequences(x)
    x = pad_sequences(x,maxlen=maxlen)
    if(new_model.predict(x)>=0.5).astype(int) == [[1]]:
        return 'True News'
    else:
        return 'False News'

#print(test('Karnataka Chief Minister Basavaraj Bommai on Wednesday dropped big hint at issuing fresh set of COVID guidelines as the COVID cases are on the rise in certain areas of the state. It was reported that the state government may also impose fresh restrictions if the cases continue to rise. However, the chief minister said decision regarding the guidelines and restrictions will be taken in a state cabinet meeting on Thursday.'))
