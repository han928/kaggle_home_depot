from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile, StringIO, requests
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.ensemble import RandomForestRegressor
import nltk.tokenize as tk
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
from nltk.stem.wordnet import WordNetLemmatizer
wordnet = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
import collections


#establish vocabulary set after stemming
def vocab_set(df,col):
    docs = [w.lower() for w in df[col]]
    tokens =[word_tokenize(content) for content in docs]
    docs_stem = [[stemmer.stem(word) for word in words] for words in tokens]
    vocab = set()
    [[vocab.add(token) for token in tokens] for tokens in docs_stem]
    return vocab

#functions to find correct spelling from existing sets
def edits1(word):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
    inserts    = [a + c + b     for a, b in s for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words):
    return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)


#correct_spelling for words in bad_dic
def correct_spelling(word):
    if word in bad_dic:
        return bad_dic[word]
    else:
        return word

#function to clean strings
def str_stem(s):
    if isinstance(s, str):
        s = s.decode('ISO-8859-1').encode('ascii', 'ignore').lower()
        s = s.replace(".",". ")
        s = s.replace(". 0",".0")
        s = s.replace(". 1",".1")
        s = s.replace(". 2",".2")
        s = s.replace(". 3",".3")
        s = s.replace(". 4",".4")
        s = s.replace(". 5",".5")
        s = s.replace(". 6",".6")
        s = s.replace(". 7",".7")
        s = s.replace(". 8",".8")
        s = s.replace(". 9",".9")
        s = s.replace("  "," ")

        s = s.replace("'","in.")
        s = s.replace("inches","in.")
        s = s.replace("inch","in.")
        s = s.replace(" in ","in. ")
        s = s.replace(" in.","in.")

        s = s.replace("''","ft.")
        s = s.replace(" feet ","ft. ")
        s = s.replace("feet","ft.")
        s = s.replace("foot","ft.")
        s = s.replace(" ft ","ft. ")
        s = s.replace(" ft.","ft.")

        s = s.replace(" pounds ","lb. ")
        s = s.replace(" pound ","lb. ")
        s = s.replace("pound","lb.")
        s = s.replace(" lb ","lb. ")
        s = s.replace(" lb.","lb.")
        s = s.replace(" lbs ","lb. ")
        s = s.replace("lbs.","lb.")

        s = s.replace(" x "," xby ")
        s = s.replace("*"," xby ")
        s = s.replace(" by "," xby")
        s = s.replace("x0"," xby 0")
        s = s.replace("x1"," xby 1")
        s = s.replace("x2"," xby 2")
        s = s.replace("x3"," xby 3")
        s = s.replace("x4"," xby 4")
        s = s.replace("x5"," xby 5")
        s = s.replace("x6"," xby 6")
        s = s.replace("x7"," xby 7")
        s = s.replace("x8"," xby 8")
        s = s.replace("x9"," xby 9")
        s = s.replace("0x","0 xby ")
        s = s.replace("1x","1 xby ")
        s = s.replace("2x","2 xby ")
        s = s.replace("3x","3 xby ")
        s = s.replace("4x","4 xby ")
        s = s.replace("5x","5 xby ")
        s = s.replace("6x","6 xby ")
        s = s.replace("7x","7 xby ")
        s = s.replace("8x","8 xby ")
        s = s.replace("9x","9 xby ")

        s = s.replace(" sq ft","sq.ft. ")
        s = s.replace("sq ft","sq.ft. ")
        s = s.replace("sqft","sq.ft. ")
        s = s.replace(" sqft ","sq.ft. ")
        s = s.replace("sq. ft","sq.ft. ")
        s = s.replace("sq ft.","sq.ft. ")
        s = s.replace("sq feet","sq.ft. ")
        s = s.replace("square feet","sq.ft. ")

        s = s.replace(" gallons ","gal. ")
        s = s.replace(" gallon ","gal. ")
        s = s.replace("gallons","gal.")
        s = s.replace("gallon","gal.")
        s = s.replace(" gal ","gal. ")
        s = s.replace(" gal","gal.")

        s = s.replace("ounces","oz.")
        s = s.replace("ounce","oz.")
        s = s.replace(" oz.","oz. ")
        s = s.replace(" oz ","oz. ")

        s = s.replace("centimeters","cm.")
        s = s.replace(" cm.","cm.")
        s = s.replace(" cm ","cm. ")

        s = s.replace("milimeters","mm.")
        s = s.replace(" mm.","mm.")
        s = s.replace(" mm ","mm. ")

        s = s.replace("°","deg. ")
        s = s.replace("degrees","deg. ")
        s = s.replace("degree","deg. ")

        s = s.replace("volts","volt. ")
        s = s.replace("volt","volt. ")

        s = s.replace("watts","watt. ")
        s = s.replace("watt","watt. ")

        s = s.replace("ampere","amp. ")
        s = s.replace("amps","amp. ")
        s = s.replace(" amp ","amp. ")

        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool")
        s = s.replace("whirlpoolstainless","whirlpool stainless")

        s = s.replace("  "," ")
        s = s.replace("..",".")
        #s = (" ").join([stemmer.stem(z) for z in s.lower().split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        return s.lower()
    else:
        return "null"

#count number of timmes that str1.split() present in str2
def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

#for search query vs title and description, count number of times whole query str1 present in str2
def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt



if __name__ == '__main__':
    df_train = pd.read_csv('train-2.csv')
    df_attributes = pd.read_csv('attributes.csv')
    df_description = pd.read_csv('product_descriptions.csv')
    df_test = pd.read_csv('test-2.csv')

    df_brand = df_attributes[df_attributes.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all = pd.merge(df_all, df_description, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')

    description_set = vocab_set(df_description,'product_description')
    search_set = vocab_set(df_all,'search_term')
    #focus on alphabetical words
    search_set2 = set([x for x in search_set if x.isalpha()])
    bad_term = set.difference(search_set2,description_set)
    #find correct_spelling by editdistance
    global NWORDS = {x:2 for x in description_set}
    global bad_dic ={x:correct(x) for x in bad_term}


    df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
    df_all['search_term'] = df_all['search_term'].apply(lambda x: ' '.join(correct_spelling(item) for item in x.split()))

    df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
    df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
    df_all['brand'] = df_all['brand'].map(lambda x:str_stem(x))

    df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
    df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
    df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
    df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)

    df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']
    df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
    df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))

    df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
    df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
    df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
    df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
    df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']
    df_all['word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
    df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']
