import nltk.tokenize as tk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer



class Recycle(object):

    def __init__(self, stop_word_lang='english'):
        self.stopwords = stopwords.words(stop_word_lang)
        self.stemmer = SnowballStemmer(stop_word_lang)
        self.wordnet = WordNetLemmatizer()


    def fit(self, df, df_description, df_attributes):
        """
        df: pandas dataframe for recycling
        """

        # load dataframes
        self.df = df.copy()
        self.df_description
        self.df_attributes

        self._fix_search_term()
        self._sum_cnt_title()
        self._sum_cnt_description()

    def _sum_cnt_description(self):
        self.df['count_common_description'] = self.df.apply(self._cnt_common_wd, axis=1)

    def _cnt_common_wd(self, x):
        """
        count the number of word in common between searchfix column and description in df_description
        """
        return sum([ word in self.df_description[self.df_description['product_uid'] == x['product_uid']]\
        ['product_description'].values[0].lower() for word in x['searchfix']])



    def _sum_cnt_title(self):  # count common word
        self.df['titlefix'] = self.df['product_title'].str.lower().str.decode('ISO-8859-1').str.encode('ascii', 'ignore')
        self.df['searchfix'] = self.df['searchfix'].apply(self.str_stem)
        self.df['count_title'] = self.df.apply(lambda df: sum([word in df['titlefix'] for word in df['searchfix']]), axis=1)


    def _fix_search_term(self):
        self.df['searchfix'] = self.df['search_term'].str.lower().str.decode('ISO-8859-1').str.encode('ascii', 'ignore').str.split()\
        .apply(lambda x: [self.stemmer.stem(item) for item in x]) \
        .apply(lambda x: [self.wordnet.lemmatize(item) for item in x])
        self.df['searchfix'] = self.df['search_term'].apply(self._str_stem)



    def _str_stem(s):
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
