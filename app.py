from flask import Flask,request,jsonify

import json

from flask_cors import CORS, cross_origin
# import util


import numpy as np
app=Flask(__name__)

CORS(app)


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
zomato_real=pd.read_csv("zomato.csv")
zomato_real.head(8000) 
zomato = zomato_real[0:12000]

zomato["reviews_list"] = zomato["reviews_list"].str.lower()

import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_punctuation(text))


import nltk
# nltk.download('stopwords')
  
## Removal of Stopwords
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_stopwords(text))

## Removal of URLS
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_urls(text))

zomato[['reviews_list', 'cuisines']].sample(5)


# RESTAURANT NAMES:
restaurant_names = list(zomato['name'].unique())
def get_top_words(column, top_nu_of_words, nu_of_word):
    vec = CountVectorizer(ngram_range= nu_of_word, stop_words='english')
    bag_of_words = vec.fit_transform(column)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:top_nu_of_words]
    


import pandas

# Randomly sample 60% of your dataframe
df_percent = zomato.sample(frac=0.5)
df_percent.set_index('name', inplace=True)
indices = pd.Series(df_percent.index)

# Creating tf-idf matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
def recommend(name, cosine_similarities = cosine_similarities):
    
    # Create a list to put top restaurants
    recommend_restaurant = []
    
    # Find the index of the hotel entered
    idx = indices[indices == name].index[0]
    
    # Find the restaurants with a similar cosine-sim value and order them from bigges number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:30].index)
    # top30_indexes = list(dict.fromkeys(top30_indexes))
    # Names of the top 30 restaurants
    # for each in top30_indexes:
    #     recommend_restaurant.append(list(df_percent.index)[each])
        
    
#     # Creating the new data set to show similar restaurants
#     df_new = pd.DataFrame(columns=['cuisines', 'rate', 'approx_cost(for two people)'])
    
#     # Create the top 30 similar restaurants with some of their columns
#     for each in recommend_restaurant:
#         df_new = df_new.append(pd.DataFrame(df_percent[['cuisines','rate', 'approx_cost(for two people)']][df_percent.index == each].sample()))
    
    
# #     # Drop the same named restaurants and sort only the top 10 by the highest rating
#     df_new = df_new.drop_duplicates(subset=['cuisines','rate', 'approx_cost(for two people)'], keep=False, inplace=False)
#     # df_new = df_new.sort_values(by='rate', ascending=False).head(10)
      
    # print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(recommend_restaurant)), name))
      
    return top30_indexes

# print(recommend('Pai Vihar'))

global __restaurents
global __menuItems


with open("./hotels.json","r") as f:
    __restaurents = json.load(f)

with open("./menuItems.json","r") as f:
    __menuItems = json.load(f)

@app.route("/menuitems")
def menuitems():
   
    response = jsonify(
        __menuItems[0:20]
    )
    response.headers.add('Access-Control-Allow-Origin','*')
    return response

@app.route("/recommendlist")
def recommendedlist():
    print("success")
    key = request.args.get("key")
    # arr=[]
    response= jsonify(recommend(key))
    response.headers.add('Access-Control-Allow-Origin','*')
    
    
    return response

@app.route("/hotels")
def restaurents():
    response = jsonify(
        __restaurents
    )
    response.headers.add('Access-Control-Allow-Origin','*')
    return response
        
    

if __name__=='__main__':
    app.run()
    print("Starting Flask server for House Price Prediction")
   
