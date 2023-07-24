import os
import tweepy
import datetime
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#get the tweet pre-processor 
import preprocessor as p
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
import joblib
from pytrends.request import TrendReq
import numpy as np
import time
pytrends = TrendReq(hl='en-US', tz=360)


consumer_key = "dedrHMOrLegD5oejTi6DHvQed"
consumer_secret = "NBGs0OGOZSsIhxmE2uos5iArsBmimXbLOwSBJvp95ITgq9enk7"
access_token = "124969809-oaUqb6lKSvL0K2fP7lCrmczO3FpV9VDlgvwYNLOR"
access_token_secret = "lTATN5CiRNTpidANEW059P47y4XuMnPk2LYfcf50oSKI4"

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
tweets_list = tweepy.Cursor(api.search_30_day, label="devTwo", query="(@cz_binance OR @APompliano OR @aantonop OR @BTC_Archive OR @MMCrypto OR @NickSzabo4) (BTC OR Bitcoin) -win", fromDate = 202211200000, toDate = 202211210000).items(50)
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
    #print(text)
    # clean the tweets, remove tags, urls
    clean_text = p.clean(text)
    print(clean_text)

    if len(clean_text) > 30:

        #pre process
        pre_processed_text = preprocess_data(clean_text)
        #print(pre_processed_text)

        #keyword
        keywords = custom_kw_extractor.extract_keywords(clean_text)
        keywordList = []
        #get the keywoeds only
        df = pd.DataFrame(keywords)
        if not df.empty:
            #print(df.get(0))
            #the number of key words
            keyWordCount = df.shape[0]
            #print(keyWordCount)
            #iterate through the keywords
            myit = iter(df.get(0))
            i = 0
            while i < keyWordCount:
                a = next(myit)
                print("keyWords:- "  + str(a))
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
df.to_csv('1-tweets.csv')

datafile = '1-tweets.csv'


############################################################ calclulate #####################################################

output = []
output_2 = []

#get the file with tweets
tweets = pd.read_csv(datafile, encoding='latin1')

#get only date from datetime
tweets['Date'] = pd.to_datetime(tweets['created_at']).dt.date

#get only the needed fields
importantFields = tweets[['clean_text','keywords','Date','% Negative','% Neutral','% Positive','favourite_count','retweet_count']]
#add index
importantFields['external_id'] = range(1, len(importantFields) + 1)
importantFields['external_id'] = importantFields['external_id'].astype(str)

print("Number of tweets: ",len(tweets['text']))

#call categorizing model
categories = ['Political', 'Economic', 'Technical', 'Market', 'Other']
#print(importantFields.clean_text.shape)
t = pd.DataFrame(importantFields)
for category in categories:
    print('... Processing {}'.format(category))
    same_pip = joblib.load('C:\\Users\\Aeshana Udadeniya\\Documents\\Masters-USCS\\project\\work\\Demo\\BTC\\classifiers\\'+category + '_model.joblib')
    # compute the testing accuracy
    prediction = same_pip.predict(importantFields.clean_text)
    print("prediction :- "+ str(prediction))
    t[category] = prediction

# Export the classifier to a file
#print(t)
conditions = [
    (t['Political'] == 1),
    (t['Economic'] == 1),
    (t['Technical'] == 1),
    (t['Market'] == 1),
    (t['Other'] == 1)]
choices = ['Political', 'Economic', 'Technical', 'Market', 'Other']
t['classifications'] = np.select(conditions, choices, default='Other')

#get only the needed fields
mergedFilterd = t[['classifications','keywords','Date','% Negative','% Neutral','% Positive','favourite_count','retweet_count']]
#print(mergedFilterd)

#group by date
df_group = mergedFilterd.groupby([mergedFilterd['Date']])
uniqueDateList = mergedFilterd.Date.unique()
#print(uniqueDateList)
#iterate for each date
for date in uniqueDateList:
    eventsPerDay = df_group.get_group(date)
    eventsPerDayPerType = eventsPerDay.groupby('classifications')
    perDayTypeList = eventsPerDay.classifications.unique()
    numberOfTypes = len(perDayTypeList)

    otherCategoryScore = 0
    politicalCategoryScore = 0
    economicCategoryScore = 0
    marketCategoryScore = 0
    technicalCategoryScore = 0
    o_positive = 0
    o_negative = 0
    o_neutral = 0
    o_fav_count = 0
    o_re_tweet = 0
    p_positive = 0
    p_negative = 0
    p_neutral = 0
    p_fav_count = 0
    p_re_tweet = 0
    s_positive = 0
    s_negative = 0
    s_neutral = 0
    s_fav_count = 0
    s_re_tweet = 0
    t_positive = 0
    t_negative = 0
    t_neutral = 0
    t_fav_count = 0
    t_re_tweet = 0
    m_positive = 0
    m_negative = 0
    m_neutral = 0
    m_fav_count = 0
    m_re_tweet = 0

    #print(perDayTypeList)
    #iterate per type in a day
    for type in perDayTypeList:
        #print(date)
        #print(type)
        eventsForAType = eventsPerDayPerType.get_group(type)

        #positive = positive + eventsForAType['% Positive'].mean()
        #negative = negative + eventsForAType['% Negative'].mean()
        #neutral = neutral + eventsForAType['% Neutral'].mean()
        #fav_count = fav_count + eventsForAType['favourite_count'].mean()
        #re_tweet = re_tweet + eventsForAType['retweet_count'].mean()

        val = eventsForAType.get('keywords')
        # isolate the key-words
        keywordList = ""
        for index in val:
            keywordList = keywordList + index
        editedKeys = keywordList.replace("][",",").replace("[","").replace("]","").replace("'","")
        keyList = list(editedKeys.split(','))
        uniqueList = list(set(keyList))
        #print(uniqueList)
        time.sleep(5)
        historicaldf = pytrends.get_historical_interest(uniqueList, year_start=date.year, month_start=date.month, day_start=date.day,
                                                        hour_start=0, year_end=date.year, month_end=date.month, day_end=date.day,
                                                        hour_end=23, cat=0, geo='', gprop='', sleep=1)
        df = pd.DataFrame(historicaldf)
        scoreMean = df.mean(axis=0).mean()
        print("mean score from google:- "+ str(scoreMean))
        if type == 'Political':
            politicalCategoryScore = scoreMean
            p_positive = eventsForAType['% Positive'].mean()
            p_negative = eventsForAType['% Negative'].mean()
            p_neutral = eventsForAType['% Neutral'].mean()
            p_fav_count = eventsForAType['favourite_count'].mean()
            p_re_tweet = eventsForAType['retweet_count'].mean()
            if p_positive < p_negative:
                politicalCategoryScore = politicalCategoryScore*-1
        elif type == 'Economic':
            economicCategoryScore = scoreMean
            s_positive = eventsForAType['% Positive'].mean()
            s_negative = eventsForAType['% Negative'].mean()
            s_neutral = eventsForAType['% Neutral'].mean()
            s_fav_count = eventsForAType['favourite_count'].mean()
            s_re_tweet = eventsForAType['retweet_count'].mean()
            if s_positive < s_negative:
                economicCategoryScore = economicCategoryScore*-1
        elif type == 'Technical':
            technicalCategoryScore = scoreMean
            t_positive = eventsForAType['% Positive'].mean()
            t_negative = eventsForAType['% Negative'].mean()
            t_neutral = eventsForAType['% Neutral'].mean()
            t_fav_count = eventsForAType['favourite_count'].mean()
            t_re_tweet = eventsForAType['retweet_count'].mean()
            if t_positive < t_negative:
                technicalCategoryScore = technicalCategoryScore*-1
        elif type == 'Market':
            marketCategoryScore = scoreMean
            m_positive = eventsForAType['% Positive'].mean()
            m_negative = eventsForAType['% Negative'].mean()
            m_neutral = eventsForAType['% Neutral'].mean()
            m_fav_count = eventsForAType['favourite_count'].mean()
            m_re_tweet = eventsForAType['retweet_count'].mean()
            if m_positive < m_negative:
                marketCategoryScore = marketCategoryScore*-1
        elif type == 'Other':
            otherCategoryScore = scoreMean
            o_positive = eventsForAType['% Positive'].mean()
            o_negative = eventsForAType['% Negative'].mean()
            o_neutral = eventsForAType['% Neutral'].mean()
            o_fav_count = eventsForAType['favourite_count'].mean()
            o_re_tweet = eventsForAType['retweet_count'].mean()
            if o_positive < o_negative:
                otherCategoryScore = otherCategoryScore*-1

    line = {'Date' : date, 'Type' : perDayTypeList, 'otherCategoryScore' : otherCategoryScore, 'politicalCategoryScore' : politicalCategoryScore,
            'economicCategoryScore' : economicCategoryScore, 'technicalCategoryScore' : technicalCategoryScore, 'marketCategoryScore' : marketCategoryScore,
            'o-% Negative' : o_negative, 'o-% Neutral': o_neutral, 'o-% Positive': o_positive,
            'o-favourite_count' : o_fav_count, 'o-retweet_count' : o_re_tweet,
            'p-% Negative': p_negative, 'p-% Neutral': p_neutral, 'p-% Positive': p_positive,
            'p-favourite_count': p_fav_count, 'p-retweet_count': p_re_tweet,
            's-% Negative': s_negative, 's-% Neutral': s_neutral, 's-% Positive': s_positive,
            's-favourite_count': s_fav_count, 's-retweet_count': s_re_tweet,
            't-% Negative': t_negative, 't-% Neutral': t_neutral, 't-% Positive': t_positive,
            't-favourite_count': t_fav_count, 't-retweet_count': t_re_tweet,
            'm-% Negative': m_negative, 'm-% Neutral': m_neutral, 'm-% Positive': m_positive,
            'm-favourite_count': m_fav_count, 'm-retweet_count': t_re_tweet
            }
    line_2 = {'Date': date, 'Type': perDayTypeList, 'otherCategoryScore': otherCategoryScore,
            'politicalCategoryScore': politicalCategoryScore,
            'economicCategoryScore': economicCategoryScore, 'technicalCategoryScore': technicalCategoryScore,
            'marketCategoryScore': marketCategoryScore,
            'o-favourite_count': o_fav_count, 'o-retweet_count': o_re_tweet,
            'p-favourite_count': p_fav_count, 'p-retweet_count': p_re_tweet,
            's-favourite_count': s_fav_count, 's-retweet_count': s_re_tweet,
            't-favourite_count': t_fav_count, 't-retweet_count': t_re_tweet,
            'm-favourite_count': m_fav_count, 'm-retweet_count': m_re_tweet
            }
    output.append(line)
    output_2.append(line_2)

df = pd.DataFrame(output)
df_2 = pd.DataFrame(output_2)
df.to_csv('2-scores-btc.csv')
df_2.to_csv('3-reduced_scores-btc.csv')

###############################################################calculate##############################################################


output_3= []
output_2 = []

#get the file with tweets
datafile = '3-reduced_scores-btc.csv'
df_2 = pd.read_csv(datafile, encoding='latin1')
df_2.fillna(0, inplace=True)
o_maxValOfFavCount = df_2['o-favourite_count'].max()
o_minValOfFavCount = df_2['o-favourite_count'].min()
o_difValOfFavCount = o_maxValOfFavCount - o_minValOfFavCount
o_maxValOfRTCount = df_2['o-retweet_count'].max()
o_minValOfRTCount = df_2['o-retweet_count'].min()
o_difValOfRTCount = o_maxValOfRTCount - o_minValOfRTCount
p_maxValOfFavCount = df_2['p-favourite_count'].max()
p_minValOfFavCount = df_2['p-favourite_count'].min()
p_difValOfFavCount = p_maxValOfFavCount - p_minValOfFavCount
p_maxValOfRTCount = df_2['p-retweet_count'].max()
p_minValOfRTCount = df_2['p-retweet_count'].min()
p_difValOfRTCount = p_maxValOfRTCount - p_minValOfRTCount
s_maxValOfFavCount = df_2['s-favourite_count'].max()
s_minValOfFavCount = df_2['s-favourite_count'].min()
s_difValOfFavCount = s_maxValOfFavCount -s_minValOfFavCount
s_maxValOfRTCount = df_2['s-retweet_count'].max()
s_minValOfRTCount = df_2['s-retweet_count'].min()
s_difValOfRTCount = s_maxValOfRTCount - s_minValOfRTCount
t_maxValOfFavCount = df_2['t-favourite_count'].max()
t_minValOfFavCount = df_2['t-favourite_count'].min()
t_difValOfFavCount = t_maxValOfFavCount - t_minValOfFavCount
t_maxValOfRTCount = df_2['t-retweet_count'].max()
t_minValOfRTCount = df_2['t-retweet_count'].min()
t_difValOfRTCount = t_maxValOfRTCount - t_minValOfRTCount
m_maxValOfFavCount = df_2['m-favourite_count'].max()
m_minValOfFavCount = df_2['m-favourite_count'].min()
m_difValOfFavCount = m_maxValOfFavCount - m_minValOfFavCount
m_maxValOfRTCount = df_2['m-retweet_count'].max()
m_minValOfRTCount = df_2['m-retweet_count'].min()
m_difValOfRTCount = m_maxValOfRTCount - m_minValOfRTCount

#print(o_difValOfRTCount,o_difValOfFavCount,s_difValOfRTCount,s_difValOfFavCount)

for index, row in df_2.iterrows():
    if 'Political' in row['Type']:
        if p_difValOfFavCount == 0:
            df_2.at[index, 'p-favourite_count'] = 0
        else:
            df_2.at[index, 'p-favourite_count'] = ((row['p-favourite_count'] - p_minValOfFavCount) / p_difValOfFavCount) * 100
        if p_difValOfRTCount == 0:
            df_2.at[index, 'p-retweet_count'] = 0
        else:
            df_2.at[index, 'p-retweet_count'] = ((row['p-retweet_count'] - p_minValOfRTCount) / p_difValOfRTCount) * 100
    if 'Economic' in row['Type']:
        if s_difValOfFavCount == 0:
            df_2.at[index, 's-favourite_count'] = 0
        else:
            df_2.at[index, 's-favourite_count'] = ((row['s-favourite_count'] - s_minValOfFavCount) / s_difValOfFavCount) * 100
        if s_difValOfRTCount == 0:
            df_2.at[index, 's-retweet_count'] = 0
        else:
            df_2.at[index, 's-retweet_count'] = ((row['s-retweet_count'] - s_minValOfRTCount) / s_difValOfRTCount) * 100
    if 'Technical' in row['Type']:
        if t_difValOfFavCount == 0:
            df_2.at[index, 't-favourite_count'] = 0
        else:
            df_2.at[index, 't-favourite_count'] = ((row['t-favourite_count'] - t_minValOfFavCount) / t_difValOfFavCount) * 100
        if t_difValOfRTCount == 0:
            df_2.at[index, 't-retweet_count'] = 0
        else:
            df_2.at[index, 't-retweet_count'] = ((row['t-retweet_count'] - t_minValOfRTCount) / t_difValOfRTCount) * 100
    if 'Market' in row['Type']:
        if m_difValOfFavCount == 0:
            df_2.at[index, 'm-favourite_count'] = 0
        else:
            df_2.at[index, 'm-favourite_count'] = ((row['m-favourite_count'] - m_minValOfFavCount) / m_difValOfFavCount) * 100
        if m_difValOfRTCount == 0:
            df_2.at[index, 'm-retweet_count'] = 0
        else:
            df_2.at[index, 'm-retweet_count'] = ((row['m-retweet_count'] - m_minValOfRTCount) / m_difValOfRTCount) * 100
    if 'Other' in row['Type']:
        if o_difValOfFavCount == 0:
            df_2.at[index, 'o-favourite_count'] = 0
        else:
            df_2.at[index, 'o-favourite_count'] = ((row['o-favourite_count'] - o_minValOfFavCount)/ o_difValOfFavCount)*100
        if o_difValOfRTCount == 0:
            df_2.at[index, 'o-retweet_count'] = 0
        else:
            df_2.at[index, 'o-retweet_count'] = ((row['o-retweet_count'] - o_minValOfRTCount)/ o_difValOfRTCount)*100

    if df_2.at[index, 'otherCategoryScore'] < 0:
        finalOtherScore = ((df_2.at[index, 'otherCategoryScore']*-1 + df_2.at[index, 'o-favourite_count']+df_2.at[index, 'o-retweet_count'])/3)*-1
    else:
        finalOtherScore = ((df_2.at[index, 'otherCategoryScore'] + df_2.at[index, 'o-favourite_count']+df_2.at[index, 'o-retweet_count'])/3)
    if df_2.at[index, 'politicalCategoryScore'] < 0:
        finalPoliticalScore = ((df_2.at[index, 'politicalCategoryScore']*-1 + df_2.at[index, 'p-favourite_count']+df_2.at[index, 'p-retweet_count'])/3)*-1
    else:
        finalPoliticalScore = ((df_2.at[index, 'politicalCategoryScore'] + df_2.at[index, 'p-favourite_count']+df_2.at[index, 'p-retweet_count'])/3)
    if df_2.at[index, 'economicCategoryScore'] < 0:
        finalSocioeconomicScore = ((df_2.at[index, 'economicCategoryScore']*-1 + df_2.at[index, 's-favourite_count']+df_2.at[index, 's-retweet_count'])/3)*-1
    else:
        finalSocioeconomicScore = ((df_2.at[index, 'economicCategoryScore'] + df_2.at[index, 's-favourite_count']+df_2.at[index, 's-retweet_count'])/3)
    if df_2.at[index, 'technicalCategoryScore'] < 0:
        finalTechnicalScore = ((df_2.at[index, 'technicalCategoryScore']*-1 + df_2.at[index, 't-favourite_count']+df_2.at[index, 't-retweet_count'])/3)*-1
    else:
        finalTechnicalScore = ((df_2.at[index, 'technicalCategoryScore'] + df_2.at[index, 't-favourite_count']+df_2.at[index, 't-retweet_count'])/3)
    if df_2.at[index, 'marketCategoryScore'] < 0:
        finalMarketScore = ((df_2.at[index, 'marketCategoryScore']*-1 + df_2.at[index, 'm-favourite_count']+df_2.at[index, 'm-retweet_count'])/3)*-1
    else:
        finalMarketScore = ((df_2.at[index, 'marketCategoryScore'] + df_2.at[index, 'm-favourite_count']+df_2.at[index, 'm-retweet_count'])/3)

    line_2 = {'Date': df_2.at[index, 'Date'], 'Type': df_2.at[index, 'Type'],
              'otherCategoryScore': df_2.at[index, 'otherCategoryScore'],
              'politicalCategoryScore': df_2.at[index, 'politicalCategoryScore'],
              'economicCategoryScore': df_2.at[index, 'economicCategoryScore'],
              'technicalCategoryScore': df_2.at[index, 'technicalCategoryScore'],
              'marketCategoryScore': df_2.at[index, 'marketCategoryScore'],
              'o-tweet-score': (df_2.at[index, 'o-favourite_count']+df_2.at[index, 'o-retweet_count'])/2,
              'p-tweet-score': (df_2.at[index, 'p-favourite_count']+df_2.at[index, 'p-retweet_count'])/2,
              's-tweet-score': (df_2.at[index, 's-favourite_count']+df_2.at[index, 's-retweet_count'])/2,
              't-tweet-score': (df_2.at[index, 't-favourite_count']+df_2.at[index, 't-retweet_count'])/2,
              'm-tweet-score': (df_2.at[index, 'm-favourite_count'] + df_2.at[index, 'm-retweet_count']) / 2,
              }
    output_2.append(line_2)
    line_3 = {'Date': df_2.at[index, 'Date'], 'Type': df_2.at[index, 'Type'],
              'otherCategoryScore': finalOtherScore,
              'politicalCategoryScore': finalPoliticalScore,
              'economicCategoryScore': finalSocioeconomicScore,
              'technicalCategoryScore': finalTechnicalScore,
              'marketCategoryScore': finalMarketScore}
    output_3.append(line_3)
df_3 = pd.DataFrame(output_2)
df_4 = pd.DataFrame(output_3)
df_2.to_csv('4-scores_norm-btc.csv')
df_3.to_csv('5-scores_norm_avg-btc.csv')
df_4.to_csv('6-scores_final-btc.csv')


######################################################### combine ###########################################

priceFile = 'C:\\Users\\Aeshana Udadeniya\\Documents\\Masters-USCS\\project\\work\\Demo\\BTC\\Data\\Bitcoin Historical Data.csv'
scoreFile = '6-scores_final-btc.csv'
df_1 = pd.read_csv(priceFile, encoding='latin1')
scoreDate = pd.read_csv(scoreFile, encoding='latin1')

df_1.rename(columns = {'ï»¿"Date"':'Date'}, inplace = True)
priceDate = df_1[['Date','Change %']]
#print(priceDate)

for ind in scoreDate.index:
    #t = time.strptime(scoreDate['Date'][ind], "%m/%d/%Y")
    t = time.strptime(scoreDate['Date'][ind], "%Y-%m-%d")
    r = time.strftime("%m/%d/%Y", t)
    scoreDate['Date'][ind] = r
#print(scoreDate)

for ind in priceDate.index:
    t = time.strptime(priceDate['Date'][ind], '%b %d, %Y')
    r = time.strftime("%m/%d/%Y", t)
    priceDate['Date'][ind] = r
    priceDate['Change %'][ind] = priceDate['Change %'][ind].replace("%","")


print(priceDate)

mergedFile = pd.merge(scoreDate, priceDate, on='Date')

print("actual price change for requested date.......")
print(mergedFile)

df_3 = pd.DataFrame(mergedFile)
df_3.to_csv('7-combined_final_data.csv')
#print(type(df_3['Change %'][2]))

df_3['Change %'] = df_3['Change %'].astype(float)
df_3.loc[df_3['Change %'] <= 0, 'Change %'] = 0
df_3.loc[df_3['Change %'] > 0, 'Change %'] = 1
df_3['Change %'] = df_3['Change %'].astype(int)

df_3.to_csv('8-combined_final_data_binary.csv')

##########################################################predict####################################################3

dataset = pd.read_csv("7-combined_final_data.csv")
dataset.drop(dataset.columns[[0, 1, 2, 3]], axis=1, inplace=True)
X = dataset.drop(['Change %'], axis=1)

#SVM
model_SVR = joblib.load('../regression_models/svm_model.joblib')
Y_pred = model_SVR.predict(X)
print("Support vector machine prediction :-"+str(Y_pred))

#Random Forest Regression
model_RFR = joblib.load('../regression_models/rf_model.joblib')
Y_pred = model_RFR.predict(X)
print("Randon forest regressor prediction :-"+str(Y_pred))

#Linear Regression
model_LR = joblib.load('../regression_models/lr_model_binary.joblib')
Y_pred = model_LR.predict(X)
print("Linear regression prediction :-"+str(Y_pred))

# #SVR
# model_SVR = joblib.load('../regression_models/svm_model_binary.joblib')
# Y_pred = model_SVR.predict(X)
# print(Y_pred)
#
# #Random Forest Regression
# model_RFR = joblib.load('../regression_models/rf_model_binary.joblib')
# Y_pred = model_RFR.predict(X)
# print(Y_pred)
#
# #Linear Regression
# model_LR = joblib.load('../regression_models/lr_model_binary.joblib')
# Y_pred = model_LR.predict(X)
# print(Y_pred)
