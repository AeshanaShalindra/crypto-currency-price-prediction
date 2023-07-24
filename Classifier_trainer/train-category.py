import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
import joblib

#https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5

df = pd.read_csv("BTC-TRAIN-2.csv", encoding = "ISO-8859-1")

categories = ['Political', 'Economic', 'Technical', 'Market', 'Other']
train, test = train_test_split(df, random_state=42, test_size=0.3, shuffle=True)
X_train = train.clean_text
X_test = test.clean_text
print(X_train.shape)
print(X_test.shape)

#Define a pipeline combining a text feature extractor with multi lable classifier

# NB_pipeline = Pipeline([
#                 ('tfidf', TfidfVectorizer(stop_words=stop_words)),
#                 ('clf', OneVsRestClassifier(MultinomialNB(
#                     fit_prior=True, class_prior=None))),
#             ])
# NB_pipeline = Pipeline([
#                 ('tfidf', TfidfVectorizer(stop_words=stop_words)),
#                 ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
#             ])

NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])

t = pd.DataFrame(X_test)
for category in categories:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    NB_pipeline.fit(X_train, train[category])
    joblib.dump(NB_pipeline, category+'_model.joblib')
    # compute the testing accuracy
    prediction = NB_pipeline.predict(X_test)
    t[category] = prediction
    #print(prediction)
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    print('Test precision is {}'.format(precision_score(test[category], prediction)))
    print('Test recall - is {}'.format(recall_score(test[category], prediction)))
    print('Test F1 - is {}'.format(f1_score(test[category], prediction)))
    print('Test report - is {}'.format(classification_report(test[category], prediction)))

    conmat = confusion_matrix(test[category], prediction)
    specificity = conmat[0,0]/(conmat[0,0]+conmat[0,1])
    sensitive = conmat[1,1]/(conmat[1,0]+conmat[1,1])
    print('Test  specificity- is {}'.format(specificity))
    print('Test sensitive - is {}'.format(sensitive))
    val = np.mat(conmat)

    classnames = list(set(train[category]))

    df_cm = pd.DataFrame(

        val, index=classnames, columns=classnames,

    )

    print(df_cm)
    plt.figure()

    heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues",fmt='g')

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.title('Logistic Regression Model Results for ' + category)

    plt.show()
#print(t)
# Export the classifier to a file
