# House-Price-Prediction-App

> House Price Prediction App

Machine Learning + Streamlit Web Application

A complete end-to-end House Price Prediction System built using Python, Scikit-Learn, and Streamlit.
This project includes both the model training pipeline and the web application for generating real-time predictions.

> Overview

This application predicts the estimated price of a house based on four key features:

*Bedrooms

*Bathrooms

*House Size (sqft)

*Location Score (1–10)

The backend uses a Random Forest Regression model trained on a custom dataset, with preprocessing handled using StandardScaler.
The frontend is built with Streamlit, providing an intuitive and responsive UI for users.

> Features
Machine Learning:-

*Random Forest Regressor (200 trees)

*StandardScaler for feature normalization

*End-to-end training script (train_model.py)

*Saved model & scaler using Joblib (house_model.pkl, house_scaler.pkl)

> Streamlit Web App

*Modern, clean user interface

*Two-column input layout

*Number input & slider widgets

*Real-time predictions

*Result formatting in Indian Rupees (₹)

*Balloon animation on prediction

> Project Structure

*app.py

*train_model.py

*house_model.pkl

*house_scaler.pkl

*README.md          

> Technologies Used

*Python 3.x

*Streamlit

*Scikit-learn

*Pandas / NumPy

*Joblib

> License

This project is licensed under the MIT License.
