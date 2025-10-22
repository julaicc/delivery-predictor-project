# delivery-predictor-project
Project model to predict if deliveries are to arrive on time or late depending on multiple factors

Dataset obtained from Kaggle:
https://www.kaggle.com/datasets/prachi13/customer-analytics

First trials nad prototyping was made on "delivery_predcit_proto.ipynb".

The files "ml_utils.py" and "01_eda_model.ipynb" are the proper backend for a Streamlit dashboard to be implemented

The predictive model of choice was Random forest. The decision was made based on the fact that we had both numerical and categorical variables and non-linear interactions and because the dataset is not that large. 

