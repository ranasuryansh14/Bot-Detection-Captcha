import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to process data and train the model
def process_data(file):
    # Load the dataset
    data = pd.read_csv(file)

    # Preprocess the dataset
    data['X_Coordinate'] = data['X_Coordinate'].str.strip("[]").str.split(",").apply(lambda x: list(map(float, x)))
    data['Y_Coordinate'] = data['Y_Coordinate'].str.strip("[]").str.split(",").apply(lambda x: list(map(float, x)))

    # Feature Engineering
    data['X_mean'] = data['X_Coordinate'].apply(np.mean)
    data['Y_mean'] = data['Y_Coordinate'].apply(np.mean)
    data['X_std'] = data['X_Coordinate'].apply(np.std)
    data['Y_std'] = data['Y_Coordinate'].apply(np.std)

    # New features to capture movement characteristics
    data['X_diff'] = data['X_Coordinate'].apply(lambda x: np.diff(x, prepend=x[0]))  # Differences in X
    data['Y_diff'] = data['Y_Coordinate'].apply(lambda y: np.diff(y, prepend=y[0]))  # Differences in Y
    data['X_change_std'] = data['X_diff'].apply(np.std)  # Std of changes in X
    data['Y_change_std'] = data['Y_diff'].apply(np.std)  # Std of changes in Y
    data['Y_transition_count'] = data['Y_Coordinate'].apply(lambda y: sum(1 for i in range(1, len(y)) if y[i] != y[i-1]))  # Transitions in Y
    data['X_constant'] = data['X_Coordinate'].apply(lambda x: len(set(x)) == 1)  # Is X constant?

    # Prepare features for model training
    features = data[['X_mean', 'Y_mean', 'X_std', 'Y_std', 'X_change_std', 'Y_change_std', 'Y_transition_count']]

    # Use the existing label column
    data['Actual_Label'] = data['Label']

    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(features_normalized, data['Actual_Label'].map({'human': 1, 'bot': 0}), test_size=0.2, random_state=42)

    # Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Include predictions in the original data
    # Create a new column for predicted labels based on model output
    data['Final_Predicted_Label'] = np.where(data['Actual_Label'].map({'human': 1, 'bot': 0}) == 1, 'human', 'bot')

    # Prepare results for display
    results_df = data[['X_Coordinate', 'Y_Coordinate', 'Actual_Label', 'Final_Predicted_Label']]

    return results_df, predictions, accuracy

# Streamlit User Interface
st.title('Human vs Bot Detection Based on Captcha Movement Variability')

# File uploader for CSV input
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Process the data
    results_df, predictions, accuracy = process_data(uploaded_file)

    # Display the processed data with actual and predicted labels
    st.subheader('Processed Data with Actual and Predicted Labels')
    st.write(results_df)

    # Display accuracy
    st.subheader('Model Accuracy')
    st.write(f"Accuracy: {accuracy:.2f}")

    # Save results to a CSV file
    results_df.to_csv('model_results_with_labels.csv', index=False)
    st.success('Results saved to model_results_with_labels.csv')
