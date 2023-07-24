import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("BTC-TRAIN-2.csv", encoding = "ISO-8859-1")
categories = ['Political', 'Economic', 'Technical', 'Market', 'Other']
print(df.clean_text.shape)
t = pd.DataFrame(df)
for category in categories:
    print('... Processing {}'.format(category))
    same_pip = joblib.load(category + '_model.joblib')
    # compute the testing accuracy
    prediction = same_pip.predict(df.clean_text)
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
t['category'] = np.select(conditions, choices, default='Other')

importantFields = t[['id','clean_text','category']]
print(importantFields)
