import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split



# Paths for data.
Ravdess = "training/voice_data/audio_speech_actors_01-24/"
Crema = "training/voice_data/AudioWAV/"
Tess = "training/voice_data/TESS Toronto emotional speech set data/"
Savee = "training/voice_data/ALL/"
localData = "training/voice_data/local/"


def load_ravdess_data(Ravdess):
    ravdess_directory_list = os.listdir(Ravdess)
    file_emotion = []
    file_path = []
    for dir in ravdess_directory_list:
        actor = os.listdir(Ravdess + dir)
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            file_emotion.append(int(part[2]))
            file_path.append(Ravdess + dir + '/' + file)

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)
    Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
    return Ravdess_df

def load_crema_data(Crema):
    crema_directory_list = os.listdir(Crema)
    file_emotion = []
    file_path = []
    for file in crema_directory_list:
        file_path.append(Crema + file)
        part=file.split('_')
        if part[2] == 'SAD':
            file_emotion.append('sad')
        elif part[2] == 'ANG':
            file_emotion.append('angry')
        elif part[2] == 'DIS':
            file_emotion.append('disgust')
        elif part[2] == 'FEA':
            file_emotion.append('fear')
        elif part[2] == 'HAP':
            file_emotion.append('happy')
        elif part[2] == 'NEU':
            file_emotion.append('neutral')
        else:
            file_emotion.append('Unknown')
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Crema_df = pd.concat([emotion_df, path_df], axis=1)
    return Crema_df

def load_tess_data(Tess):
    tess_directory_list = os.listdir(Tess)
    file_emotion = []
    file_path = []
    for dir in tess_directory_list:
        actor = os.listdir(Tess + dir)
        for file in actor:
            parts = file.split('.')[0].split('_')
            if len(parts) < 3: 
                continue  # skip files without three parts in the name
            part = parts[2]
            if part == 'ps':
                file_emotion.append('surprise')
            else:
                file_emotion.append(part)
            file_path.append(Tess + dir + '/' + file)
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Tess_df = pd.concat([emotion_df, path_df], axis=1)
    return Tess_df

def load_savee_data(Savee):
    savee_directory_list = os.listdir(Savee)
    file_emotion = []
    file_path = []
    for file in savee_directory_list:
        file_path.append(Savee + file)
        part = file.split('_')[1]
        ele = part[:-6]
        if ele == 'a':
            file_emotion.append('angry')
        elif ele == 'f':
            file_emotion.append('fear')
        elif ele == 'h':
            file_emotion.append('happy')
        elif ele == 'n':
            file_emotion.append('neutral')
        elif ele == 's':
            file_emotion.append('sad')
        else:
            file_emotion.append('Unknown')
        
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Savee_df = pd.concat([emotion_df, path_df], axis=1)
    return Savee_df


# Planning to save local data files like this date_time_emotion.wav. ex: 2021-05-01_12-00-00_happy.wav

def load_local_data(localData):
    local_directory_list = os.listdir(localData)
    file_emotion = []
    file_path = []
    for file in local_directory_list:
        emotion = file.split('_')[-1].replace('.wav', '')
        file_emotion.append(emotion)
        file_path.append(localData + file)
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    local_df = pd.concat([emotion_df, path_df], axis=1)
    return local_df
 


def load_data():
    Ravdess_df = load_ravdess_data(Ravdess)
    Crema_df = load_crema_data(Crema)
    Tess_df = load_tess_data(Tess)
    Savee_df = load_savee_data(Savee)
    local_df = load_local_data(localData)
    data = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df, local_df], axis=0)
    data.to_csv("training/csv/voice_data.csv")


# data augmentation
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=0.8)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7): #,
    return librosa.effects.pitch_shift(data,sr=sampling_rate,n_steps=0.7)



def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally

    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data,sample_rate)
    result = np.vstack((result, res2)) # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch,sample_rate)
    result = np.vstack((result, res3)) # stacking vertically

    return result

def create_feature_df(voiceDatasetPath):
    df = pd.read_csv(voiceDatasetPath)
    df = df[df.Emotions != 'Unknown']
    X, Y = [], []
    for path, emotion in zip(df.Path, df.Emotions):
        feature = get_features(path)
        for ele in feature:
            X.append(ele)
            # appending emotion 3 times since implemented 3 augmentation techniques on each audio file.
            Y.append(emotion)
    Features = pd.DataFrame(X)
    Features['labels'] = Y
    Features.to_csv('training/information/features.csv', index=False)
    Features.head()



def main():
    # Load the data
    load_data()

    # Specify the path to your dataset
    voiceDatasetPath = "training/csv/voice_data.csv"

    # Create the feature data frame
    create_feature_df(voiceDatasetPath)

