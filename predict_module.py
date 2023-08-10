import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import speech_recognition as sr
from deep_translator import GoogleTranslator
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import pickle

# Load the model
def load_audio_model():
    model = load_model('models/best_test.h5')
    return model

def load_svm_model():
    # Load the saved model
    clf = joblib.load('models/svm_model.pkl')

    # Load the training data to fit the vectorizer
    df = pd.read_csv('models/sentence.csv')  
    X_train = df['phrase']

    # Create the vectorizer and fit it with the training data
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)
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
    with sr.AudioFile(audio) as source:
      r.adjust_for_ambient_noise(source)
      audio = r.record(source)
      text = r.recognize_google(audio, language='si-LK')
    translated_text = GoogleTranslator(source='si', target='en').translate(text)
    return translated_text,text

def predict_audio_emotion(path, model):
    correlation_selected_indices = [13, 14, 15, 16, 17, 18, 21, 30, 31, 32, 33, 40, 41, 42, 43, 44, 45, 46,
                                    47, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67, 68, 69, 70, 71, 72,
                                    73, 76, 77, 78, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 92, 93, 94, 95,
                                    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                    111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
                                    125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
                                    139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
                                    153, 154, 155, 156, 157, 158, 159, 160, 161]
    test = get_features(path)
    #test = test[correlation_selected_indices]
    test = np.reshape(test, (1, 162, 1))
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
    prediction = predict_text_department(sentence, clf, vectorizer)

    return decoded_predictions[0], prediction, sentence,sinhala_sentence
