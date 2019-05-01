import numpy as np
import pandas as pd
import nltk
from keras import callbacks
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
from keras.utils import to_categorical
import random
from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense,Dropout,Embedding,LSTM
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
lemmatizer = WordNetLemmatizer()
set_random_seed(123)
random.seed(123)
train= pd.read_csv("train.tsv", sep="\t")
test = pd.read_csv("test.tsv", sep="\t")


def clean_sentences(df):
    reviews = []

    for sent in tqdm(df['Phrase']):
        # remove html content
        review_text = BeautifulSoup(sent).get_text()

        # remove non-alphabetic characters
        review_text = re.sub("[^a-zA-Z]", " ", review_text)

        # tokenize the sentences
        words = word_tokenize(review_text.lower())

        # lemmatize each word to its lemma
        lemma_words = [lemmatizer.lemmatize(i) for i in words]

        reviews.append(lemma_words)

    return (reviews)
train_sentences = clean_sentences(train)
test_sentences = clean_sentences(test)
print(len(train_sentences))
print(len(test_sentences))
target=train.Sentiment.values
y_target=to_categorical(target)
num_classes=y_target.shape[1]
X_train,X_val,y_train,y_val=train_test_split(train_sentences,y_target,test_size=0.2,stratify=y_target)
unique_words = set()
len_max = 0

for sent in tqdm(X_train):

    unique_words.update(sent)

    if (len_max < len(sent)):
        len_max = len(sent)

# length of the list of unique_words gives the no of unique words
print(len(list(unique_words)))
print(len_max)
tokenizer = Tokenizer(num_words=len(list(unique_words)))
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(test_sentences)

#padding done to equalize the lengths of all input reviews. LSTM networks needs all inputs to be same length.
#Therefore reviews lesser than max length will be made equal using extra zeros at end. This is padding.
X_train = sequence.pad_sequences(X_train, maxlen=len_max)
X_val = sequence.pad_sequences(X_val, maxlen=len_max)
X_test = sequence.pad_sequences(X_test, maxlen=len_max)
print(X_train.shape,X_val.shape,X_test.shape)

#Model using Keras LSTM
model=Sequential()

model.add(Embedding(len(list(unique_words)),300,input_length=len_max))
model.add(LSTM(128,dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.005),metrics=['accuracy'])
model.summary()
tensorboard = TensorBoard(log_dir="logs/{}",histogram_freq=0, write_graph=True, write_images=True)
history=model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=1, batch_size=256, verbose=1,callbacks=[tensorboard])
#model = load_model('./checkpoint_model1.h5')

# visualize model architecture
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

y_pred=model.predict_classes(X_test)
print(y_pred)
sub_file = pd.read_csv('sampleSubmission.csv',sep=',')
sub_file.Sentiment=y_pred
#sub_file.to_csv('Submission.csv',index=False)

