from monkeylearn import MonkeyLearn
import numpy as np
from sklearn.pipeline import Pipeline
import joblib
import time
from pytrends.request import TrendReq
import pandas as pd
pytrends = TrendReq(hl='en-US', tz=360)

output = []
output_2 = []

#get the file with tweets
datafile = 'Data/output-xrp-full-pat-year.csv'
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
print(importantFields.clean_text.shape)
t = pd.DataFrame(importantFields)
for category in categories:
    print('... Processing {}'.format(category))
    same_pip = joblib.load('classifiers/'+category + '_model.joblib')
    # compute the testing accuracy
    prediction = same_pip.predict(importantFields.clean_text)
    print(prediction)
    t[category] = prediction

# Export the classifier to a file
print(t)
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
print(mergedFilterd)

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
        print(date)
        print(type)
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
        time.sleep(1)
        historicaldf = pytrends.get_historical_interest(uniqueList, year_start=date.year, month_start=date.month, day_start=date.day,
                                                        hour_start=0, year_end=date.year, month_end=date.month, day_end=date.day,
                                                        hour_end=12, cat=0, geo='', gprop='', sleep=1)
        df = pd.DataFrame(historicaldf)
        scoreMean = df.mean(axis=0).mean()
        print(scoreMean)
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
df.to_csv('Outputs/scores-xrp.csv')
df_2.to_csv('Outputs/reduced_scores-xrp.csv')

o_maxValOfFavCount = df_2['o-favourite_count'].max()
o_minValOfFavCount = df_2['o-favourite_count'].min()
o_maxValOfRTCount = df_2['o-retweet_count'].max()
o_minValOfRTCount = df_2['o-retweet_count'].min()
p_maxValOfFavCount = df_2['p-favourite_count'].max()
p_minValOfFavCount = df_2['p-favourite_count'].min()
p_maxValOfRTCount = df_2['p-retweet_count'].max()
p_minValOfRTCount = df_2['p-retweet_count'].min()
s_maxValOfFavCount = df_2['s-favourite_count'].max()
s_minValOfFavCount = df_2['s-favourite_count'].min()
s_maxValOfRTCount = df_2['s-retweet_count'].max()
s_minValOfRTCount = df_2['s-retweet_count'].min()
t_maxValOfFavCount = df_2['t-favourite_count'].max()
t_minValOfFavCount = df_2['t-favourite_count'].min()
t_maxValOfRTCount = df_2['t-retweet_count'].max()
t_minValOfRTCount = df_2['t-retweet_count'].min()
m_maxValOfFavCount = df_2['m-favourite_count'].max()
m_minValOfFavCount = df_2['m-favourite_count'].min()
m_maxValOfRTCount = df_2['m-retweet_count'].max()
m_minValOfRTCount = df_2['m-retweet_count'].min()

print(o_maxValOfFavCount,o_minValOfFavCount,o_minValOfRTCount,o_maxValOfRTCount)