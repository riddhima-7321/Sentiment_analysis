from flask import Flask,render_template,request
import pickle
import numpy as np
import tensorflow as tf
import string
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
dlmodel = tf.keras.models.load_model('dlmodel.keras')
model = Word2Vec.load('word2vec.model')
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_sentiment():
    Text = request.form.get('text')
    Text = Text.lower()
    Text = Text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    Text = nltk.word_tokenize(Text)
    Text = [word for word in Text if word.lower() not in stop_words]
    Text = ' '.join(Text)
    word_vectors = [model.wv[word] for word in Text.split() if word in model.wv]
    word_vectors = np.array(word_vectors)
    result = dlmodel.predict(word_vectors.reshape(1, -1, 100))
    result=np.argmax(result,axis=1)
    return str(result)

if __name__ == '__main__':
    app.run(debug=True)
   