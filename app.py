import gradio as gr
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the dataset and get the column names
dataset = pd.read_csv('Covid Dataset.csv')
columns_to_drop = ['Wearing Masks', 'Sanitization from Market']
dataset = dataset.drop(columns_to_drop, axis=1)
column_names = dataset.columns.tolist()

# Define a function to make predictions using the selected models
def predict_covid(*symptoms):
    # Convert None values to Falses
    symptoms = [False if symptom is None else symptom for symptom in symptoms]
    
    if sum(symptoms) == 0:
        return "COVID-19 Negative"
    
    # Load the saved models
    model_logreg = joblib.load('logreg_model.h5')
    model_rf_classifier = joblib.load('rf_classifier_model.h5')
    model_dt_classifier = joblib.load('dt_classifier_model.h5')
    model_knn_classifier = joblib.load('knn_classifier_model.h5')
    model_svm_classifier = joblib.load('svm_classifier_model.h5')
    model_ann_classifier = joblib.load('ann_model.h5')
    voting_classifier = joblib.load('voting_classifier_model.h5')
    stacking_classifier = joblib.load('stacking_classifier_model.h5')

    # Prepare the input data
    label_encoder = LabelEncoder()
    input_data = pd.DataFrame([list(symptoms)], columns=column_names[:-1])
    encoded_input_data = input_data.copy()
    for column in encoded_input_data.columns:
        if encoded_input_data[column].dtype == object:
            encoded_input_data[column] = label_encoder.transform(encoded_input_data[column])

    # Make predictions using the selected models
    logreg_prediction = int(model_logreg.predict(encoded_input_data)[0])
    rf_prediction = int(model_rf_classifier.predict(encoded_input_data)[0])
    dt_prediction = int(model_dt_classifier.predict(encoded_input_data)[0])
    knn_prediction = int(model_knn_classifier.predict(encoded_input_data)[0])
    svm_prediction = int(model_svm_classifier.predict(encoded_input_data)[0])
    ann_prediction = int(model_ann_classifier.predict(encoded_input_data)[0])
    voting_prediction = int(voting_classifier.predict(encoded_input_data)[0])
    stacking_prediction = int(stacking_classifier.predict(encoded_input_data)[0])

    # Determine the overall prediction
    prediction = 1 if sum([logreg_prediction, rf_prediction, dt_prediction, knn_prediction,
                          svm_prediction, ann_prediction, voting_prediction, stacking_prediction]) >= 4 else 0

    # Return the prediction
    return "COVID-19 Positive" if prediction == 1 else "COVID-19 Negative"

# Create a list of checkboxes for the dataset columns
checkboxes = [gr.inputs.Checkbox(label=column_name) for column_name in column_names[:-1]]

# Create the input interface with the checkboxes
inputs = checkboxes

# Create the output interface with the predicted labels
outputs = gr.outputs.Label(num_top_classes=1, label="COVID-19 Status")
title = "COVID-19 Detection"
description = "Select your symptoms and contact history to check if you have COVID-19"
final_model = gr.Interface(fn=predict_covid, inputs=inputs, outputs=outputs, title=title, description=description)

# Create the Gradio app
if __name__ == '__main__':
    final_model.launch(inline=False)