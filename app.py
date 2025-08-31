import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv('heart.csv')

# Features and label
X = df.drop('target', axis=1)
y = df['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model & scaler
joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")






# Prediction function
def check_abnormal_parameters(data):
    abnormal = {}
    if data['cp'] in [2, 3]: abnormal['cp'] = data['cp']
    if data['trestbps'] > 130: abnormal['trestbps'] = data['trestbps']
    if data['chol'] > 240: abnormal['chol'] = data['chol']
    if data['fbs'] == 1: abnormal['fbs'] = 'Fasting > 120 mg/dl'
    if data['restecg'] != 0: abnormal['restecg'] = data['restecg']
    if data['thalach'] < 120: abnormal['thalach'] = data['thalach']
    if data['exang'] == 1: abnormal['exang'] = 'Yes'
    if data['oldpeak'] >= 2.0: abnormal['oldpeak'] = data['oldpeak']
    if data['slope'] != 1: abnormal['slope'] = data['slope']
    if data['ca'] > 0: abnormal['ca'] = data['ca']
    if data['thal'] in [2, 3]: abnormal['thal'] = data['thal']
    return abnormal

# Streamlit app
st.title("💓 Heart Disease Prediction App")
st.write("This app predicts whether a patient has heart disease based on their health parameters.")

# Sidebar inputs
st.sidebar.header("Patient Parameters")
age = st.sidebar.slider('Age', 20, 80, 50)
sex = st.sidebar.selectbox('Sex', ['Female', 'Male'])
cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3])
trestbps = st.sidebar.slider('Resting Blood Pressure', 80, 200, 120)
chol = st.sidebar.slider('Cholesterol', 100, 400, 200)
fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
restecg = st.sidebar.selectbox('Resting ECG', [0, 1, 2])
thalach = st.sidebar.slider('Max Heart Rate', 60, 210, 150)
exang = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.0, 1.0, 0.1)
slope = st.sidebar.selectbox('Slope', [0, 1, 2])
ca = st.sidebar.slider('Major Vessels (ca)', 0, 4, 0)
thal = st.sidebar.selectbox('Thal', [1, 2, 3])
family_history = st.sidebar.selectbox('Family History', [0, 1])

# Predict button
if st.sidebar.button("Predict"):
    input_data = {
        'age': age,
        'sex': 1 if sex == 'Male' else 0,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    values = np.array(list(input_data.values())).reshape(1, -1)
    values_scaled = scaler.transform(values)
    prediction = model.predict(values_scaled)[0]

    if prediction == 1:
        st.error("💔 Heart Disease Detected")
        abnormal = check_abnormal_parameters(input_data)
        if abnormal:
            st.warning("⚠️ Abnormal Parameters:")
            for key, val in abnormal.items():
                st.write(f"- **{key}**: {val}")
        else:
            st.info("✅ No abnormal parameters detected.")
    else:
        st.success("❤️ No Heart Disease Detected")

# Show metrics
st.subheader("📊 Model Performance")
y_pred = model.predict(X_test)
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"]))

# Confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Heart Disease"],
            yticklabels=["No Disease", "Heart Disease"], ax=ax)
st.pyplot(fig)

# Correlation heatmap
st.subheader("Feature Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
st.pyplot(fig2)


# Histograms of all features
st.subheader("Feature Histograms")
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))  # Adjust rows/cols based on features
axes = axes.flatten()

for i, col in enumerate(df.columns):
    axes[i].hist(df[col], bins=20, color='skyblue', edgecolor='black')
    axes[i].set_title(col)

# Remove any empty subplots if df has fewer columns
for j in range(len(df.columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
st.pyplot(fig)
