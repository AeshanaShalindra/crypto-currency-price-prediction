import time
import pandas as pd

output_3= []
output_2 = []

#get the file with tweets
datafile = 'Outputs/reduced_scores-btc.csv'
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

print(o_difValOfRTCount,o_difValOfFavCount,s_difValOfRTCount,s_difValOfFavCount)

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
df_2.to_csv('Outputs/scores_norm-btc.csv')
df_3.to_csv('Outputs/scores_norm_avg-btc.csv')
df_4.to_csv('Outputs/scores_final-btc.csv')