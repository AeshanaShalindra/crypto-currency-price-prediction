import time
import pandas as pd

priceFile = 'C:\\Users\\Demo\\XRP\\Data\\XRP Historical Data 2.csv'
scoreFile = 'Outputs/scores_final-xrp.csv'
df_1 = pd.read_csv(priceFile, encoding='latin1')
scoreDate = pd.read_csv(scoreFile, encoding='latin1')

df_1.rename(columns = {'ï»¿"Date"':'Date'}, inplace = True)
priceDate = df_1[['Date','Change %']]
print(priceDate)

for ind in scoreDate.index:
    #t = time.strptime(scoreDate['Date'][ind], "%m/%d/%Y")
    t = time.strptime(scoreDate['Date'][ind], "%Y-%m-%d")
    r = time.strftime("%m/%d/%Y", t)
    scoreDate['Date'][ind] = r
print(scoreDate)

for ind in priceDate.index:
    t = time.strptime(priceDate['Date'][ind], '%b %d, %Y')
    r = time.strftime("%m/%d/%Y", t)
    priceDate['Date'][ind] = r
    priceDate['Change %'][ind] = priceDate['Change %'][ind].replace("%","")


print(priceDate)

mergedFile = pd.merge(scoreDate, priceDate, on='Date')

print(mergedFile)

df_3 = pd.DataFrame(mergedFile)
df_3.to_csv('Outputs/combined_final_data-xrp.csv')
print(type(df_3['Change %'][2]))

df_3['Change %'] = df_3['Change %'].astype(float)
df_3.loc[df_3['Change %'] <= 0, 'Change %'] = 0
df_3.loc[df_3['Change %'] > 0, 'Change %'] = 1
df_3['Change %'] = df_3['Change %'].astype(int)

df_3.to_csv('Outputs/combined_final_data_2-xrp.csv')