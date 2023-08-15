import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import speech_recognition as sr
from deep_translator import GoogleTranslator
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import pickle

# Load the model
def load_audio_model():
    model = load_model('models/best_model.h5')
    return model

def load_svm_model():
    # Load the saved model
    clf,vectorizer = joblib.load('models/svm_model.pkl')

    return clf, vectorizer

# Audio features extraction
def extract_features(data,sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr))

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result

def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)
    return result

def sinhala_audio_to_text(audio):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio) as source:
            r.adjust_for_ambient_noise(source)
        audio = r.record(source)
        text = r.recognize_google(audio, language='si-LK')
        translated_text = GoogleTranslator(source='si', target='en').translate(text)
        return translated_text,text
    except ValueError as e:
        print(f"An error occurred while reading the audio file: {e}")
        return None, None

def predict_audio_emotion(path, model):
    with open('models/selected_indices.txt', 'r') as file:
        correlation_selected_indices= list(map(int, file.read().strip().split(',')))
    test = get_features(path)
    test = test[correlation_selected_indices]
    test = np.reshape(test, (1, 120, 1))
    predictions = model.predict(test)
    return predictions


def decode_predictions(predictions):
    # Load the reverse mapping
    with open('models/reverse_mapping.pkl', 'rb') as f:
        reverse_mapping = pickle.load(f)
    decoded_predictions = [reverse_mapping[np.argmax(prediction)] for prediction in predictions]
    return decoded_predictions

def predict_text_department(sentence, clf, vectorizer):
    new_data = pd.DataFrame({'phrase': [sentence]})
    X_new_transformed = vectorizer.transform(new_data['phrase'])
    prediction = clf.predict(X_new_transformed)[0]
    return prediction

def predict(path):
    model = load_audio_model()
    clf, vectorizer = load_svm_model()

    predictions = predict_audio_emotion(path, model)
    decoded_predictions = decode_predictions(predictions)

    sinhala_sentence ,sentence = sinhala_audio_to_text(path)

    if sinhala_sentence is None or sentence is None:
        print("An error occurred while processing the audio file. Prediction cannot be made.")
        return None, None, None, None
    
    prediction = predict_text_department(sentence, clf, vectorizer)

    return decoded_predictions[0], prediction, sentence,sinhala_sentence
