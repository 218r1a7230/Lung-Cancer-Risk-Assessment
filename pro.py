import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

  # Update this path if needed
data = pd.read_csv('C:\\Users\\prane\\Downloads\\survey-lung-cancer.csv')

# Strip any spaces and standardize column names
data.columns = data.columns.str.strip().str.replace(' ', '_').str.upper()

# Convert categorical variables to numerical values
label_encoder_gender = LabelEncoder()
label_encoder_smoking = LabelEncoder()
label_encoder_yellow_fingers = LabelEncoder()
label_encoder_anxiety = LabelEncoder()
label_encoder_peer_pressure = LabelEncoder()
label_encoder_chronic_disease = LabelEncoder()
label_encoder_fatigue = LabelEncoder()
label_encoder_allergy = LabelEncoder()
label_encoder_wheezing = LabelEncoder()
label_encoder_alcohol_consuming = LabelEncoder()
label_encoder_coughing = LabelEncoder()
label_encoder_shortness_of_breath = LabelEncoder()
label_encoder_swallowing_difficulty = LabelEncoder()
label_encoder_chest_pain = LabelEncoder()
label_encoder_lung_cancer = LabelEncoder()

data['GENDER'] = label_encoder_gender.fit_transform(data['GENDER'])
data['SMOKING'] = label_encoder_smoking.fit_transform(data['SMOKING'])
data['YELLOW_FINGERS'] = label_encoder_yellow_fingers.fit_transform(data['YELLOW_FINGERS'])
data['ANXIETY'] = label_encoder_anxiety.fit_transform(data['ANXIETY'])
data['PEER_PRESSURE'] = label_encoder_peer_pressure.fit_transform(data['PEER_PRESSURE'])
data['CHRONIC_DISEASE'] = label_encoder_chronic_disease.fit_transform(data['CHRONIC_DISEASE'])
data['FATIGUE'] = label_encoder_fatigue.fit_transform(data['FATIGUE'])
data['ALLERGY'] = label_encoder_allergy.fit_transform(data['ALLERGY'])
data['WHEEZING'] = label_encoder_wheezing.fit_transform(data['WHEEZING'])
data['ALCOHOL_CONSUMING'] = label_encoder_alcohol_consuming.fit_transform(data['ALCOHOL_CONSUMING'])
data['COUGHING'] = label_encoder_coughing.fit_transform(data['COUGHING'])
data['SHORTNESS_OF_BREATH'] = label_encoder_shortness_of_breath.fit_transform(data['SHORTNESS_OF_BREATH'])
data['SWALLOWING_DIFFICULTY'] = label_encoder_swallowing_difficulty.fit_transform(data['SWALLOWING_DIFFICULTY'])
data['CHEST_PAIN'] = label_encoder_chest_pain.fit_transform(data['CHEST_PAIN'])
data['LUNG_CANCER'] = label_encoder_lung_cancer.fit_transform(data['LUNG_CANCER'])

# Separate features and target variable
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Streamlit application
st.title("Lung Cancer Prediction")

st.sidebar.header("Patient Data")
gender = st.sidebar.selectbox("Gender", ("M", "F"))
age = st.sidebar.slider("Age", 1, 100, 25)
smoking = st.sidebar.selectbox("Smoking", ("NO", "YES"))
yellow_fingers = st.sidebar.selectbox("Yellow Fingers", ("NO", "YES"))
anxiety = st.sidebar.selectbox("Anxiety", ("NO", "YES"))
peer_pressure = st.sidebar.selectbox("Peer Pressure", ("NO", "YES"))
chronic_disease = st.sidebar.selectbox("Chronic Disease", ("NO", "YES"))
fatigue = st.sidebar.selectbox("Fatigue", ("NO", "YES"))
allergy = st.sidebar.selectbox("Allergy", ("NO", "YES"))
wheezing = st.sidebar.selectbox("Wheezing", ("NO", "YES"))
alcohol_consuming = st.sidebar.selectbox("Alcohol Consuming", ("NO", "YES"))
coughing = st.sidebar.selectbox("Coughing", ("NO", "YES"))
shortness_of_breath = st.sidebar.selectbox("Shortness of Breath", ("NO", "YES"))
swallowing_difficulty = st.sidebar.selectbox("Swallowing Difficulty", ("NO", "YES"))
chest_pain = st.sidebar.selectbox("Chest Pain", ("NO", "YES"))

new_data = {
    'GENDER': gender,
    'AGE': age,
    'SMOKING': smoking,
    'YELLOW_FINGERS': yellow_fingers,
    'ANXIETY': anxiety,
    'PEER_PRESSURE': peer_pressure,
    'CHRONIC_DISEASE': chronic_disease,
    'FATIGUE': fatigue,
    'ALLERGY': allergy,
    'WHEEZING': wheezing,
    'ALCOHOL_CONSUMING': alcohol_consuming,
    'COUGHING': coughing,
    'SHORTNESS_OF_BREATH': shortness_of_breath,
    'SWALLOWING_DIFFICULTY': swallowing_difficulty,
    'CHEST_PAIN': chest_pain
}

# Convert the input to a DataFrame
input_df = pd.DataFrame([new_data])

# Encode categorical features using the same encoder fitted on training data
input_df['GENDER'] = label_encoder_gender.transform(input_df['GENDER'])
input_df['SMOKING'] = label_encoder_smoking.transform(input_df['SMOKING'])
input_df['YELLOW_FINGERS'] = label_encoder_yellow_fingers.transform(input_df['YELLOW_FINGERS'])
input_df['ANXIETY'] = label_encoder_anxiety.transform(input_df['ANXIETY'])
input_df['PEER_PRESSURE'] = label_encoder_peer_pressure.transform(input_df['PEER_PRESSURE'])
input_df['CHRONIC_DISEASE'] = label_encoder_chronic_disease.transform(input_df['CHRONIC_DISEASE'])
input_df['FATIGUE'] = label_encoder_fatigue.transform(input_df['FATIGUE'])
input_df['ALLERGY'] = label_encoder_allergy.transform(input_df['ALLERGY'])
input_df['WHEEZING'] = label_encoder_wheezing.transform(input_df['WHEEZING'])
input_df['ALCOHOL_CONSUMING'] = label_encoder_alcohol_consuming.transform(input_df['ALCOHOL_CONSUMING'])
input_df['COUGHING'] = label_encoder_coughing.transform(input_df['COUGHING'])
input_df['SHORTNESS_OF_BREATH'] = label_encoder_shortness_of_breath.transform(input_df['SHORTNESS_OF_BREATH'])
input_df['SWALLOWING_DIFFICULTY'] = label_encoder_swallowing_difficulty.transform(input_df['SWALLOWING_DIFFICULTY'])
input_df['CHEST_PAIN'] = label_encoder_chest_pain.transform(input_df['CHEST_PAIN'])

# Normalize the data using the same scaler fitted on training data
input_df = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_df)
prediction_label = label_encoder_lung_cancer.inverse_transform(prediction)

st.write(f"Lung Cancer Prediction: {prediction_label[0]}")

# Display accuracy and other metrics
st.write(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
st.write('Classification Report:')
st.text(classification_report(y_test, y_pred))
st.write('Confusion Matrix:')
st.write(confusion_matrix(y_test, y_pred))
