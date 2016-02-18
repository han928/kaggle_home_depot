
import pandas as pd
import numpy as np
import scipy as sp
import zipfile
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
wordnet = WordNetLemmatizer()
stemmer = SnowballStemmer('english')


def str_stem(s):
    if isinstance(s, str):
        s = s.lower()
        s = s.replace("'","in.") # character
        s = s.replace("inches","in.") # whole word
        s = s.replace("inch","in.") # whole word
        s = s.replace(" in ","in. ") # no period
        s = s.replace(" in.","in.") # prefix space

        s = s.replace("''","ft.") # character
        s = s.replace(" feet ","ft. ") # whole word
        s = s.replace("feet","ft.") # whole word
        s = s.replace("foot","ft.") # whole word
        s = s.replace(" ft ","ft. ") # no period
        s = s.replace(" ft.","ft.") # prefix space

        s = s.replace(" pounds ","lb. ") # character
        s = s.replace(" pound ","lb. ") # whole word
        s = s.replace("pound","lb.") # whole word
        s = s.replace(" lb ","lb. ") # no period
        s = s.replace(" lb.","lb.")
        s = s.replace(" lbs ","lb. ")
        s = s.replace("lbs.","lb.")

        s = s.replace("*"," xby ")
        s = s.replace(" by"," xby")
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

        s = s.replace(" sq ft","sq.ft. ")
        s = s.replace("sq ft","sq.ft. ")
        s = s.replace("sqft","sq.ft. ")
        s = s.replace(" sqft ","sq.ft. ")
        s = s.replace("sq. ft","sq.ft. ")
        s = s.replace("sq ft.","sq.ft. ")
        s = s.replace("sq feet","sq.ft. ")
        s = s.replace("square feet","sq.ft. ")

        s = s.replace(" gallons ","gal. ") # character
        s = s.replace(" gallon ","gal. ") # whole word
        s = s.replace("gallons","gal.") # character
        s = s.replace("gallon","gal.") # whole word
        s = s.replace(" gal ","gal. ") # character
        s = s.replace(" gal","gal") # whole word

        s = s.replace(" ounces","oz.")
        s = s.replace(" ounce","oz.")
        s = s.replace("ounce","oz.")
        s = s.replace(" oz ","oz. ")

        s = s.replace(" centimeters","cm.")
        s = s.replace(" cm.","cm.")
        s = s.replace(" cm ","cm. ")

        s = s.replace(" milimeters","mm.")
        s = s.replace(" mm.","mm.")
        s = s.replace(" mm ","mm. ")

        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool")
        s = s.replace("whirlpoolstainless","whirlpool stainless")

        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        #volts, watts, amps
        return s.lower()
    else:
        return "null"







if __name__ == '__main__':
    df = pd.read_csv(zipfile.ZipFile('data/train_home_depot.zip', mode='r').open('train.csv'))
    df['search_term'] = df['search_term'].apply(str_stem)

    df['searchfix'] = df['search_term'].str.lower().str.decode('ISO-8859-1').str.encode('ascii', 'ignore').str.split()\
    .apply(lambda x: [stemmer.stem(item) for item in x]) \
    .apply(lambda x: [wordnet.lemmatize(item) for item in x])

    rating_df = pd.DataFrame(columns=['search_term', 'product_uid','relevance'])


    # code to generate df of long data
    for row in df.iterrows():
        for wd in row[1]['searchfix']:
            rating_df = rating_df.append({'search_term':wd,'product_uid':row[1]['product_uid'], 'relevance':row[1]['relevance']}, ignore_index=True)
