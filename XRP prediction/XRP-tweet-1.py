import os
import tweepy
import datetime
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#get the tweet pre-processor 
import preprocessor as p

#from monkeylearn import MonkeyLearn


#important libraries for preprocessing using NLTK
import nltk
nltk.download
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.tokenize import TweetTokenizer

#from chart_studio import plotly as py
import plotly as py
import plotly.graph_objs as go

from gensim import corpora, models
import logging
import tempfile
from nltk.corpus import stopwords
from string import punctuation
from collections import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
import yake

from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360)


consumer_key = #add consumer key
consumer_secret = #add secret key
access_token = #add access token
access_token_secret = #add secret token

# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# Setting your access token and secret
auth.set_access_token(access_token, access_token_secret)
# Creating the API object while passing in auth information
api = tweepy.API(auth) 

# The search term you want to find
query = "Bitcoin"
# Language code (follows ISO 639-1 standards)
language = "en"

#keyword params  Unsupervised approach Corpus-Independent Domain and Language Independent Single-Document -https://github.com/LIAAD/yake
max_ngram_size = 3
deduplication_threshold = 0.1
numOfKeywords = 2

custom_kw_extractor = yake.KeywordExtractor(lan="en", n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)


today = datetime.date.today()
yesterday= today - datetime.timedelta(days=365)

sid_obj = SentimentIntensityAnalyzer()
tweets_list = tweepy.Cursor(api.search_full_archive, label="DevThree", query="(@cz_binance OR @APompliano OR @aantonop OR @BTC_Archive OR @MMCrypto OR @NickSzabo4) (XRP OR ripple) -win", fromDate = 202110201212).items(5)
#tweets_list = tweepy.Cursor(api.search_tweets, q="(from:VitalikButerin OR from:IncomeSharks OR from:im_goomba OR from:rektcapital OR from:scottmelker OR from:CryptoMichNL OR from:DanAshCrypto OR from:VentureCoinist) #BTC since:" + str(yesterday)+ " until:" + str(today),tweet_mode='extended', lang='en', count=10000).items(10000)
#tweets_list = tweepy.Cursor(api.search_30_day, query="(@cz_binance OR @APompliano OR @aantonop OR @BTC_Archive OR @MMCrypto OR @NickSzabo4) (BTC OR Bitcoin OR bitcoin) -win", fromDate=202210171516, label = 'devTwo').items(1000)
#tweets_list = tweepy.Cursor(api.search_30_day, query="(@cz_binance OR @APompliano OR @aantonop OR @BTC_Archive OR @MMCrypto OR @NickSzabo4) (XRP OR Ripple) -win", fromDate=202210171516, label = 'devTwo').items(1000)
#tweets_list = tweepy.Cursor(api.search_30_day, query="(@cz_binance OR @APompliano OR @aantonop OR @BTC_Archive OR @MMCrypto OR @NickSzabo4) (Doge OR DOGE OR dogecoin) -win", fromDate=202210171516, label = 'devTwo').items(1000)

#tweets_list = tweepy.Cursor(api.search_30_day, query="#XRP -#crypto -#BTC -#whale -#Live OR from:elonmusk",result_type='popular', fromDate=202208201503, label = 'devTwo').items(10)
print("Number of tweets: ",tweets_list)
output = []

def preprocess_data(data):
 #Removes Numbers
 data = data.replace('\d+', '')
 lower_text = data.lower()
 lemmatizer = nltk.stem.WordNetLemmatizer()
 w_tokenizer =  TweetTokenizer()
 
 def lemmatize_text(text):
  return [(lemmatizer.lemmatize(w)) for w \
                       in w_tokenizer.tokenize((text))]
 def remove_punctuation(words):
  new_words = []
  for word in words:
      new_word = re.sub(r'[^\w\s]', '', (word))
      if new_word != '':
         new_words.append(new_word)
  return new_words
 words = lemmatize_text(lower_text)
 words = remove_punctuation(words)
 return pd.DataFrame(words)

for tweet in tweets_list:
    text = tweet._json["text"]
    print(text)
    # clean the tweets, remove tags, urls
    clean_text = p.clean(text)
    print(clean_text)

    if len(clean_text) > 30:

        #pre process
        pre_processed_text = preprocess_data(clean_text)
        print(pre_processed_text)

        #keyword
        keywords = custom_kw_extractor.extract_keywords(clean_text)
        keywordList = []
        #get the keywoeds only
        df = pd.DataFrame(keywords)
        if not df.empty:
            #print(df.get(0))
            #the number of key words
            keyWordCount = df.shape[0]
            print(keyWordCount)
            #iterate through the keywords
            myit = iter(df.get(0))
            i = 0
            while i < keyWordCount:
                a = next(myit)
                print(a)
                keywordList.append(a)
                i += 1

            favourite_count = tweet.favorite_count
            retweet_count = tweet.retweet_count
            created_at = tweet.created_at
            user = tweet.user.screen_name
            user_location = tweet.user.location
            user_description = tweet.user.description
            user_verified = tweet.user.verified

            #get sentiment
            sentiment_dict = sid_obj.polarity_scores(clean_text)


            line = {'user' : user, 'user_location' : user_location, 'user_description' : user_description, 'user_verified' : user_verified, 'text' : text, 'clean_text' : clean_text, 'keywords' : keywordList,
            '% Negative' : sentiment_dict['neg']*100, '% Neutral' : sentiment_dict['neu']*100, '% Positive' : sentiment_dict['pos']*100, 'favourite_count' : favourite_count, 'retweet_count' : retweet_count, 'created_at' : created_at}
            output.append(line)

df = pd.DataFrame(output)
df.to_csv('Outputs/output.csv')

datafile = 'Outputs/output.csv'

tweets = pd.read_csv(datafile, encoding='latin1')
tweets = tweets.assign(Time=pd.to_datetime(tweets.created_at))

print("Number of tweets: ",len(tweets['text']))
tweets.head(5)

tweets['Time'] = pd.to_datetime(tweets['Time'], format='%y-%m-%d %H:%M:%S')
tweetsT = tweets['Time']

trace = go.Histogram(
    x=tweetsT,
    marker=dict(
        color='blue'
    ),
    opacity=0.75
)

layout = go.Layout(
    title='Tweet Activity Over Years',
    height=450,
    width=1200,
    xaxis=dict(
        title='Month and year'
    ),
    yaxis=dict(
        title='Tweet Quantity'
    ),
    bargap=0.2,
)

data = [trace]

fig = go.Figure(data=data, layout=layout)
py.offline.plot(fig)

corpus=[]
a=[]
for i in range(len(tweets['text'])):
        a=tweets['text'][i]
        corpus.append(a)
        
corpus[0:10]

TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# removing common words and tokenizing
list1 = ['RT','rt']
stoplist = stopwords.words('english') + list(punctuation) + list1

texts = [[word for word in str(document).lower().split() if word not in stoplist] for document in corpus]

dictionary = corpora.Dictionary(texts)
dictionary.save(os.path.join(TEMP_FOLDER, 'elon.dict'))  # store the dictionary, for future reference

#print(dictionary)
#print(dictionary.token2id)
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'elon.mm'), corpus)  # store to disk, for later use

tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

corpus_tfidf = tfidf[corpus]  # step 2 -- use the model to transform vectors

total_topics = 5
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)
corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
#Show first n important word in the topics:
lda.show_topics(total_topics,5)

data_lda = {i: OrderedDict(lda.show_topic(i,25)) for i in range(total_topics)}
#data_lda
df_lda = pd.DataFrame(data_lda)
df_lda = df_lda.fillna(0).T
print(df_lda.shape)

g=sns.clustermap(df_lda.corr(), center=0, standard_scale=1, cmap="RdBu", metric='cosine', linewidths=.75, figsize=(15, 15))
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()
