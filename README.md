# crypto-currency-price-prediction
A complete workflow to use machine learning with tweets , google trends and past price data to predict the future price changes in crypto


![image](https://github.com/AeshanaShalindra/crypto-currency-price-prediction/assets/25500157/086a011a-b743-4cf9-8e73-4bd407554013)

As shown in the above diagram, tweets which have been collected will be categorised in to 5 categories, namely political, economic, tech,marhet and other.

The classifier_trainer file contains the training and testing components for ML classifier models 

#How to run

1. First select one of the prediction file , BTC or XRP

2. Add your twitter developer account keys to btc-tweet-1.py file

3. Run the python scripts in the following order

eg:- selected BTC
btc-tweet-1  (gathers tweets for today)
btc-categorize-2 (categorises tweets in to the 5 categories)
btc-calculate-3 (gets google trend values)
btc-combine-4 (normalizes the scores)
btc-prediction-5 (gives price prediction)
btc-prediction-binary-6 (gives increase or decrease signals)

5. Check the outputs in the "Outputs file"

Please find the complete article about this in 
https://medium.com/@aeshanashalindra/how-can-machine-learning-help-us-predict-financial-markets-9afd282e0624
