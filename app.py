from flask import Flask, render_template, request
import keras

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import pandas as pd

port = 12345
def project_id():
    import json
    import os
    info = json.load(open(os.path.join(os.environ['HOME'], ".smc", "info.json"), 'r'))
    return info['project_id']

base_url = "/%s/port/%s/" % (project_id(), port)
static_url = "/%s/port/%s/static" % (project_id(), port)
app = Flask(__name__, static_url_path=static_url)

model = keras.models.load_model('')

langs_dict = {'ara' : 'Arabic', 'eng':'English', 'spa':"Spanish", 'fra':"French", 
             'deu':'German','ita':'Italian',  'vie' :'Vietnamese', 'cmn':'Mandarin Chinese', 'nld':'Dutch',
             'por':'Portuguese', 'rom':'Romany'}

# Reading Series
train_max = pd.read_csv('train_max.csv', index_col = 0, squeeze = True)
train_min = pd.read_csv('train_min.csv', index_col = 0, squeeze = True)


vocab = {}
with open('vocab.pk', 'rb') as fin:
    vocab = pickle.load(fin)

vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,3), vocabulary=vocab)
with open('vectorizer.pk', 'rb') as fin:
    vectorizer = pickle.load(fin)
    
feature_names = []
with open('feature_names.pk', 'rb') as fin:
    feature_names = pickle.load(fin)

label_encoder = LabelEncoder()
with open('label_encoder.pk', 'rb') as fin:
    label_encoder = pickle.load(fin)


    
@app.route(base_url)
def home():
    name = "Language Identifier - Universal"
    return render_template('Home.html', name=name)


@app.route(base_url+"/identify_get", methods=['POST', 'GET'])
def get_language():
    sentence = request.args.get('msg')
    return predict(sentence)

def predict(sentence):
    X = vectorizer.fit_transform([sentence])
    X = pd.DataFrame(data=X.toarray(),columns=feature_names)
    X = (X - train_min)/(train_max-train_min)
    language_probabilities = model.predict(X)
    language_index = np.argmax(language_probabilities, axis = -1)
    language_name = label_encoder.inverse_transform(language_index)
    
    return langs_dict[language_name[0]]


if __name__ == "__main__":
    # you will need to change code.ai-camp.org to other urls if you are not running on the coding center.
    print("Try to open\n\n    https://cocalc3.ai-camp.org" + base_url + '\n\n')
    app.run(host = '0.0.0.0', port = port, debug=True)
    
    import sys; sys.exit(0)
