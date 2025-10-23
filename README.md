# Delivery Predictor Project
Project model to predict if deliveries are to arrive on time or late depending on multiple factors

Dataset obtained from Kaggle:
https://www.kaggle.com/datasets/prachi13/customer-analytics

First trials nad prototyping was made on "delivery_predcit_proto.ipynb".

The files "ml_utils.py" and "01_eda_model.ipynb" are the proper backend for a Streamlit dashboard to be implemented

The predictive model of choice was Random forest. The decision was made based on the fact that we had both numerical and categorical variables and non-linear interactions and because the dataset is not that large. 

Outcome:
              precision    recall  f1-score   support

           0       0.58      0.69      0.63       895
           1       0.76      0.65      0.70      1305

    accuracy                           0.67      2200
   macro avg       0.67      0.67      0.66      2200
weighted avg       0.68      0.67      0.67      2200

ROC AUC: 0.7496556005051477

What does this mean?
0 means delivery was on time, 1 means it was late.

From that table we can conclude that when the model says “on time,” it’s right 58% of the time, and it manages to catch 69% of all real on-time deliveries. 
When the model says “late,” it’s right 76% of the time, and it correctly identifies 65% of the truly late shipments.

CONCLUSION: The model is better at predicting late deliveries.

ROC-AUC can be interpreted as following:
Perfect model: 1.0, Random guessing: 0.5
Therefore, 0.75 is a good and usaful predictive power, considering specially this is a first model. 
Further adding features could improve the model. 



