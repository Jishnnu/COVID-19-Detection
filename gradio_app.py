import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
import mlxtend
import joblib
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import save_model, load_model
from PIL import Image
from gradio import components as gr_components

# Load the dataset and get the column names
dataset = pd.read_csv('Covid_Dataset.csv')
columns_to_drop = ['Wearing Masks', 'Sanitization from Market']
dataset = dataset.drop(columns_to_drop, axis=1)
column_names = dataset.columns.tolist()

# Load the text-based model
model_logreg = joblib.load('logreg_model.h5')
model_rf_classifier = joblib.load('rf_classifier_model.h5')
model_dt_classifier = joblib.load('dt_classifier_model.h5')
model_knn_classifier = joblib.load('knn_classifier_model.h5')
model_svm_classifier = joblib.load('svm_classifier_model.h5')
model_ann_classifier = joblib.load('ann_model.h5')
voting_classifier = joblib.load('voting_classifier_model.h5')
stacking_classifier = joblib.load('stacking_classifier_model.h5')

# CustomScaleLayer definition
class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomScaleLayer, self).__init__()

    def call(self, inputs):        
        return inputs * 2

# Register the custom layer within the custom object scope
tf.keras.utils.get_custom_objects()['CustomScaleLayer'] = CustomScaleLayer

# Load the image-based models with the custom object scope
inception_model = tf.keras.models.load_model('xray_inception_model.h5', compile=False)
resnet_model = tf.keras.models.load_model('xray_resnet_model.h5', compile=False)
densenet_model = tf.keras.models.load_model('xray_densenet_model.h5', compile=False)
vgg19_model = tf.keras.models.load_model('xray_vgg19_model.h5', compile=False)
efficientnet_model = tf.keras.models.load_model('xray_efficientnet_model.h5', compile=False)
mobilenet_model = tf.keras.models.load_model('xray_mobilenet_model.h5', compile=False)

# Define the ensemble models
ensemble_models = {
    'InceptionResnetV2': inception_model,
    'ResNet50': resnet_model,
    'DenseNet121': densenet_model,
    'VGG19': vgg19_model,
    'EfficientNetV2B3': efficientnet_model,
    'MobileNetV2': mobilenet_model
}

class_labels = ['COVID-19', 'Normal', 'Viral Pneumonia']
class_index_mapping = {i: label for i, label in enumerate(class_labels)}

# Define a dictionary with custom weightage for each symptom
custom_symptom_weightage = {
    "Breathing Problem": 1.0,
    "Fever": 1.0,
    "Dry Cough": 1.0,
    "Sore throat": 1.0,
    "Running Nose": 1.0,
    "Asthma": 1.0,
    "Chronic Lung Disease": 1.0,
    "Headache": 1.0,
    "Heart Disease": 0.7,
    "Diabetes": 1.0,
    "Hyper Tension": 1.0,
    "Fatigue": 0.9,
    "Gastrointestinal": 0.7,
    "Abroad Travel": 0.5,
    "Contact with COVID Patient": 1.0,
    "Attended Large Gathering": 0.5,
    "Visited Public Exposed Places": 0.5,
    "Family Working in Public Exposed Places": 0.5,
}

# Create a bar chart to represent symptom weightage
def show_symptom_weightage(symptoms, weights):
    selected_symptoms = [symptom for symptom, weight in zip(symptoms, weights) if weight > 0]

    # Check if any symptom is selected by the user
    if len(selected_symptoms) == 0:
        # If no symptom is selected, return None to indicate no bar graph should be displayed
        return None

    selected_weights = [weight for weight in weights if weight > 0]
    plt.figure(figsize=(10, 6))
    plt.bar(selected_symptoms, selected_weights)
    plt.xlabel("Symptom")
    plt.ylabel("Weight")
    plt.title("Weightage/Importance of Selected Symptoms")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot as an image and return the file path
    file_path = tempfile.NamedTemporaryFile(suffix=".png").name
    plt.savefig(file_path)
    plt.close()
    return file_path

# Create a heat map to represent image-based model predictions
def show_heat_map(image, prediction_probabilities):
    plt.figure(figsize=(8, 6))
    sns.heatmap(prediction_probabilities, annot=True, cmap="YlGnBu", xticklabels=class_labels, yticklabels=False)
    plt.title("Image-based Model Prediction Heatmap")
    plt.xlabel("Prediction")
    plt.ylabel("Model")
    plt.tight_layout()
    # Save the plot as an image and return the file path
    file_path = tempfile.NamedTemporaryFile(suffix=".png").name
    plt.savefig(file_path)
    plt.close()
    return file_path
    
# Apply the same rescaling as in the model
def rescale_images(img):
    return img / 127.5 - 1

def is_scan_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < 15
    
# Define a function to preprocess the image
def preprocess_image(image):        
    image = image.copy()  
    image = Image.fromarray(image)  # Convert the array to PIL Image
    image = image.resize((128, 128))         
    image = np.array(image)
    image = rescale_images(image)      
    return image
    
# Define a function to make predictions using the selected models
def predict_covid(*args):    
    symptoms = [False if symptom is None else symptom for symptom in args[:-1]] # Convert None values to False
    image = args[-1]

    if not any(symptoms) and image is None:
        return "COVID-19 Negative", "100%", "COVID-19 Negative", "100%", "No inputs defined", None, None

    if any(symptoms) and image is None:
        return "NA", "NA", "NA", "NA", "Please select your Symptoms and provide your Chest X-Ray Scan", None, None    
        
    # Prepare the input data for the text-based model
    label_encoder = LabelEncoder()
    input_data = pd.DataFrame([list(symptoms)], columns=column_names[:-1])
    encoded_input_data = input_data.copy()
    for column in encoded_input_data.columns:
        if encoded_input_data[column].dtype == object:
            encoded_input_data[column] = label_encoder.transform(encoded_input_data[column])

    # Make predictions using the text-based model
    logreg_prediction = int(model_logreg.predict(encoded_input_data)[0])
    rf_prediction = int(model_rf_classifier.predict(encoded_input_data)[0])
    dt_prediction = int(model_dt_classifier.predict(encoded_input_data)[0])
    knn_prediction = int(model_knn_classifier.predict(encoded_input_data)[0])
    svm_prediction = int(model_svm_classifier.predict(encoded_input_data)[0])
    ann_prediction = int(model_ann_classifier.predict(encoded_input_data)[0])
    voting_prediction = int(voting_classifier.predict(encoded_input_data)[0])
    stacking_prediction = int(stacking_classifier.predict(encoded_input_data)[0])

    # Get the probabilities for text-based models
    logreg_prob_positive = model_logreg.predict_proba(encoded_input_data)[0][1]
    rf_prob_positive = model_rf_classifier.predict_proba(encoded_input_data)[0][1]
    dt_prob_positive = model_dt_classifier.predict_proba(encoded_input_data)[0][1]
    knn_prob_positive = model_knn_classifier.predict_proba(encoded_input_data)[0][1]    
    ann_prob_positive = model_ann_classifier.predict_proba(encoded_input_data)[0][1]
    if model_svm_classifier.probability:
        svm_prob_positive = model_svm_classifier.predict_proba(encoded_input_data)[0][1]
    else:
        svm_prob_positive = model_svm_classifier.decision_function(encoded_input_data)[0]


    # Determine the overall prediction from the text-based model
    positive_predictions = sum([logreg_prediction, rf_prediction, dt_prediction,
                                knn_prediction, svm_prediction, ann_prediction])

    text_prediction = "COVID-19 Positive" if positive_predictions >= 4 else "COVID-19 Negative"

    # Determine the overall prediction from the text-based model
    if text_prediction is None:
        text_prediction = "NA"
        text_prob_positive = 0.0

    # Calculate the average probability of positive prediction    
    weights = [custom_symptom_weightage[symptom] if symptom in custom_symptom_weightage else 0.0 for symptom in column_names[:-1]]
    probabilities = [logreg_prob_positive, rf_prob_positive, dt_prob_positive,
                     knn_prob_positive, svm_prob_positive, ann_prob_positive]

    # Manually calculate the weighted average
    total_weighted_prob = sum(prob * weight for prob, weight in zip(probabilities, weights))
    total_weight = sum(weights)
    text_prob_positive = (total_weighted_prob / total_weight) * 100 if total_weight != 0 else 0.0

    # Visualize the symptom weightage/importance    
    symptom_weightage_img = show_symptom_weightage(column_names[:-1], weights)    
    
    # Prepare the input data for the image-based model
    image_prediction = None
    confidence = None
    reason = None

    if image is not None:
        is_blurry = is_scan_blurry(image)
        if is_blurry:
            return (
                "Invalid X-Ray Scan",
                "NA",
                "Invalid X-Ray Scan",
                "NA",
                "Your X-Ray scan does not meet the required standards. Please ensure that your scans are not blurry, pixelated, or disfigured",
                None,
                None
            )
        
        image = preprocess_image(image)

        # Make predictions using the ensemble models
        ensemble_predictions = []
        for model_name, model in ensemble_models.items():
            prediction = model.predict(image[np.newaxis, ...])[0]
            probabilities = prediction / np.sum(prediction)  
            ensemble_predictions.append((prediction, probabilities))
    
        # Calculate the average prediction from the ensemble models
        avg_ensemble_prob = np.mean([probabilities for _, probabilities in ensemble_predictions], axis=0)
        ensemble_prediction = np.argmax(avg_ensemble_prob)
        image_prediction = class_index_mapping[ensemble_prediction]

        # Calculate the confidence level of the prediction
        confidence = np.max(avg_ensemble_prob) * 100

        # Visualize the heatmap
        heatmap_img = show_heat_map(image, [prediction for prediction, _ in ensemble_predictions])
        
        # Provide reasoning for the prediction
        if ensemble_prediction == 0 and text_prediction == "COVID-19 Positive":
            reason = "Your X-Ray scan and symptoms provide conclusive indications of COVID-19" 
        
        elif text_prediction == "COVID-19 Positive" and ensemble_prediction != 0:
            if text_prob_positive > confidence:                
                reason = "Your symptoms provide conclusive indications of COVID-19"
                
            else:
                reason = "Your X-Ray scan provides no conclusive indications of COVID-19"
        
        elif text_prediction == "COVID-19 Negative" and ensemble_prediction != 0:
                reason = "Your X-Ray scan and symptoms provide no conclusive indications of COVID-19"
        
        elif ensemble_prediction == 0 and text_prediction == "COVID-19 Negative":
            if text_prob_positive > confidence:                
                reason = "Your symptoms provide no conclusive indications of COVID-19"
                
            else:
                reason = "Your X-Ray scan provides conclusive indications of COVID-19" 
            
        else:
            reason = "Your X-Ray scan provides no conclusive indications of COVID-19"
    
    else:
        reason = "Your symptoms provide conclusive indications of your diagnosis"

    # Ensure that confidence and other variables have a valid value
    confidence = 0.0 if confidence is None else confidence
    heatmap_img = "" if heatmap_img is None else heatmap_img
    symptom_weightage_img = "" if symptom_weightage_img is None else symptom_weightage_img
        
    return text_prediction, f"{text_prob_positive:.2f}%", image_prediction, f"{confidence:.2f}%", reason, heatmap_img, symptom_weightage_img

# Create the input and output components
symptom_components = [gr_components.Checkbox(label=label) for label in column_names[:-1]]
image_component = gr_components.Image()
output_components = [
    gr_components.Label(label="Prediction based on Symptoms"),
    gr_components.Label(label="Symptom Confidence (%)"),
    gr_components.Label(label="Prediction based on X-Ray Scan"),
    gr_components.Label(label="X-Ray Scan Confidence (%)"),
    gr_components.Textbox(label="Final Prediction"),
    gr_components.Image(label="X-Ray Prediction Heatmap"),
    gr_components.Image(label="Weightage assigned to Symptoms")
]

# Create the interface and launch
iface = gr.Interface(
    fn=predict_covid,
    inputs=symptom_components + [image_component],
    outputs=output_components,
    title="COVID-19 Detection", 
    description="Select the symptom(s) from the list and upload the X-Ray image of your chest to test for COVID-19. Test results will be available instantly. Please wait as it may take some time to process your input and make a prediction."
)

if __name__ == '__main__':
    iface.launch(inline=False)
