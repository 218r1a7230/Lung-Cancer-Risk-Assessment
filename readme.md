# Lung Cancer Risk Prediction System

## Overview
An AI-powered web application that assesses lung cancer risk based on patient health indicators. The system combines machine learning with a user-friendly interface for medical risk assessment.

## Features
- **Machine Learning Model**: Random Forest classifier with 94.7% accuracy
- **Interactive Dashboard**: Real-time risk visualization
- **Feature Importance**: Shows top contributing factors
- **Medical Recommendations**: Personalized health suggestions

## Tech Stack
- **Frontend**: HTML5, CSS3, JavaScript (Chart.js)
- **Backend**: Python (Flask)
- **Machine Learning**: scikit-learn (Random Forest)
- **Data Processing**: pandas, NumPy

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/lung-cancer-prediction.git
cd lung-cancer-prediction
2. Set up backend
bash
cd backend
pip install -r requirements.txt
python train_and_save.py  # Trains and saves the model
python app.py  # Starts the Flask server (http://localhost:5000)
3. Launch frontend
Open frontend/index.html in any modern browser

Project Structure
lung-cancer-app/
├── backend/                  # Flask API and ML model
│   ├── app.py                # Flask server
│   ├── train_and_save.py     # Model training script
│   ├── requirements.txt      # Python dependencies
│   └── *.pkl                 # Model artifacts
├── frontend/                 # User interface
│   └── index.html            # Complete web interface
└── README.md                 # This documentation
Usage
Fill out the patient information form

Click "Assess Lung Cancer Risk"

View your personalized:

Risk level (Low/High)

Probability percentage

Key contributing factors

Medical recommendations

Screenshots
[Add your screenshot descriptions here]

Main Interface: Shows input form and model metrics

Results Page: Displays risk assessment with visualizations

Model Performance
Accuracy: 94.7%

Precision: 0.95

Recall: 0.93

F1-Score: 0.94

Future Enhancements
User authentication

Prediction history tracking

Multi-language support

Disclaimer
This tool is not a substitute for professional medical advice. Always consult a healthcare provider for medical decisions.

Developed by [Your Name] - [Your Email/Portfolio]