import tensorflow as tf
## Load the data to get started
import pandas as pd
import numpy as np
import tensorflow as tf
import spacy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import keras
import string
import pickle


# Shamelessly copied from http://flask.pocoo.org/docs/quickstart/

from flask import Flask, render_template,request
app = Flask(__name__)




#loading model 
model = tf.keras.models.load_model('E:\\iti++\\AIM task\\project\\static\\my_keras_model.h5')




# preprocessing text
arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ه", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def nospecial(text):
    
	text = re.sub("[a-zA-Z0-9]+", "",text)
	return text


def preprocess_text(tweet): 

    #remove emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub(r'', tweet)
    
  
    #remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    tweet = re.sub(p_tashkeel,"", tweet)

    #Replace @username with empty string
    tweet = re.sub('@[^\s]+', ' ', tweet)
    
    #Convert www.* or https?://* to " "
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet)
    
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    # remove punctuations
    tweet= remove_punctuations(tweet)
    
    # normalize the tweet
     #tweet= normalize_arabic(tweet)
    
    # remove repeated letters
    tweet=remove_repeating_char(tweet)

    #remove non arabic words
    tweet = nospecial(tweet)

    #trim    
    tweet = tweet.strip()

    #   #text stemming
    # from nltk.stem.isri import ISRIStemmer
    # st = ISRIStemmer()
    # ISRIStemmer().suf32(tweet)
    
    return tweet

def embedding(tweet):

    with open('E:\\iti++\\AIM task\\project\\static\\tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle) 
    
        X = tokenizer.texts_to_sequences(tweet)
        X = pad_sequences(X,maxlen=61)

        return X


def prediction(tweet):
    label_encodding = {
    0:"AE" , 
    1:"BH" ,
    2:"DZ",
    3:"EG",
    4:"IQ",
    5:"JO",
    6:"KW",
    7:"LB",
    8:"LY",
    9:"MA",
    10:"OM",
    11:"PL",
    12:"QA",
    13:"SA",
    14:"SD",
    15:"SY",
    16:"TN",
    17:"YE"}
    tweet = preprocess_text(tweet)
    tweet = embedding([tweet])
    print(model.predict(tweet))
    pred = model.predict(tweet)
    classes_x = np.argmax(pred,axis=1)
    print(classes_x)
    return (label_encodding.get(classes_x[0]))

# model.predict

# prediction("طعميه")


@app.route('/')
def home():
    return render_template('index.html')
    # return prediction("تمام")

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        glon = request.form['tweet']
        glon = prediction(glon)
        return render_template('index.html', prediction_text=glon)

if __name__ == '__main__':
    app.run()
