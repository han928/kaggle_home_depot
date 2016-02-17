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



# open files  all file inside data/ folder and in zip format
z = zipfile.ZipFile('data/train_home_depot.zip')
df_train = pd.read_csv(z.open('train.csv'))
z2 = zipfile.ZipFile('data/attributes.csv.zip')
df_attributes = pd.read_csv(z2.open('attributes.csv'))
z3 = zipfile.ZipFile('data/product_descriptions.csv.zip')
df_description = pd.read_csv(z3.open('product_descriptions.csv'))

z4 = zipfile.ZipFile('data/test.csv.zip')
df_test = pd.read_csv(z4.open('test.csv'))

def data_cleaning(df_org):

    df = df_org.copy()

    # stem and lematize search term
    df['searchfix'] = df['search_term'].str.lower().str.decode('ISO-8859-1').str.encode('ascii', 'ignore').str.split()\
    .apply(lambda x: [stemmer.stem(item) for item in x]) \
    .apply(lambda x: [wordnet.lemmatize(item) for item in x])

    # count common word between search term and title
    df['titlefix'] = df['product_title'].str.lower().str.decode('ISO-8859-1').str.encode('ascii', 'ignore')
    def sum_title(df):
        return sum([word in df['titlefix'] for word in df['searchfix']])
    df['count_title'] = df.apply(sum_title, axis=1)


    # count common word between search term and df_description
    def num_word_descrp(x):
        """
        count the number of word in common between searchfix column and description in df_description
        """
        return sum([ word in df_description[df_description['product_uid'] == x['product_uid']]['product_description'].values[0].lower() for word in x['searchfix']])

    df['count_common_description'] = df.apply(num_word_descrp, axis=1)


    # count common word between search term and brand name fro df_attributes
    brandnames = df_attributes[df_attributes.name == "MFG Brand Name"][['product_uid', 'value']]
    brandnames.index = brandnames['product_uid']
    brandnames.value = brandnames.value.str.lower()

    df_search2 = df.join(brandnames, on='product_uid', lsuffix='l', rsuffix='r').drop(['product_uidl', 'product_uidr'], axis=1)
    df_search2.fillna('NaN', inplace=True)
    df_search2['cnt'] = df_search2.apply(lambda row: sum(row.value.find(word) > 0 for word in row.search_term), axis=1)


    return df_search2

df_train_cleaned = data_cleaning(df_train)
df_test_cleaned = data_cleaning(df_test)

RF_mod = RandomForestRegressor(50)
# cross_val_score(RF_mod, X, y, cv=5, scoring='r2')



def modeling(estimator, df, df_test, col, submission=False):
    X = df[col]
    y = df['relevance']
    X_test = df_test[col]
    estimator.fit(X, y)

    result = estimator.predict(X_test)

    output_df= pd.DataFrame(df_test['id'],columns=['id',"relevance"])
    output_df['relevance'] = result
    output_df['relevance'] = output_df['relevance'].apply(lambda x: 3 if x>3 else x)
    output_df['relevance'] = output_df['relevance'].apply(lambda x: 1 if x<1 else x)


    if submission:
        output_df.to_csv('submission.csv',index=False)

    return estimator

    
col=['count_title','count_common_description','cnt']
mod = modeling(RF_mod, df_train_cleaned, df_test_cleaned, col)
