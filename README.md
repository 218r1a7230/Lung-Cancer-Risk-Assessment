# Lung Cancer Risk Prediction using Random Forest and Streamlit

This project uses a Random Forest Classifier to predict the risk of lung cancer based on patient data collected from a survey.  A Streamlit application provides a user-friendly interface for inputting patient information and receiving predictions.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)

## Introduction

Lung cancer is a leading cause of death worldwide. Early detection is crucial for improving patient outcomes. This project aims to develop a machine learning model that can predict the likelihood of lung cancer based on various risk factors.  The model is deployed using Streamlit, making it accessible and easy to use.

## Dataset

The dataset used in this project is the "Survey Lung Cancer" dataset, available on Kaggle: [https://www.kaggle.com/datasets/arshid/survey-lung-cancer](https://www.kaggle.com/datasets/arshid/survey-lung-cancer).
This dataset contains 309 instances with 16 features, including:

* Gender
* Age
* Smoking status
* Yellow fingers
* Anxiety
* Peer pressure
* Chronic disease
* Fatigue
* Allergy
* Wheezing
* Alcohol consuming
* Coughing
* Shortness of breath
* Swallowing difficulty
* Chest pain
* Lung Cancer (target variable)

## Model

A Random Forest Classifier was chosen for this prediction task due to its ability to handle both categorical and numerical features, as well as its robustness to overfitting.  The model was trained using Scikit-learn. 
