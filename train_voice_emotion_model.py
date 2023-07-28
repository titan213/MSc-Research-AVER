import pandas as pd
import numpy as np
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import io
import urllib
import base64

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, regularizers

import extract_voice_features as evf

def load_data():
    evf.main() # create features.csv
    features=pd.read_csv('training/information/features.csv')
    X = features.iloc[: ,:-1].values
    Y = features['labels'].values
    return X, Y

def save_pickle_objects(label_encoder, reverse_mapping):
    with open('training/information/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('training/information/reverse_mappling.pkl', 'wb') as f:
        pickle.dump(reverse_mapping, f)
    label_mapping = dict(enumerate(label_encoder.classes_))
    with open('training/information/label_mapping.pkl', 'wb') as f:
        pickle.dump(label_mapping, f)

def select_features(X, Y):
    correlation_selector = SelectKBest(score_func=f_classif, k=120)
    selected_features_correlation = correlation_selector.fit_transform(X, Y)
    return selected_features_correlation

def split_data(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X,Y, random_state=42, shuffle=True)
    return x_train, x_test, y_train, y_test

def encode_labels(y_train, y_test):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_train_encoded = y_train_encoded.reshape(-1, 1)
    y_test_encoded = y_test_encoded.reshape(-1, 1)
    onehot_encoder = OneHotEncoder()
    y_train_onehot = onehot_encoder.fit_transform(y_train_encoded).toarray()
    y_test_onehot = onehot_encoder.transform(y_test_encoded).toarray()
    reverse_mapping = {encoded_label: original_label for encoded_label, original_label in zip(y_train_encoded.flatten(), y_train)}

    print('Number of classes:', y_train_onehot.shape[1])

    save_pickle_objects(label_encoder, reverse_mapping) # Save mappings

    return y_train_onehot, y_test_onehot, label_encoder, onehot_encoder

def expand_dims(x_train, x_test):
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    return x_train, x_test

def define_model():
    model = tf.keras.Sequential([
        layers.Conv1D(64, 3, padding='same', activation='relu', input_shape=(120, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.5),
        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(1),
        layers.Dropout(0.5),
        layers.Conv1D(512, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(1),
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(8, activation='softmax')
    ])
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    checkpoint_filepath = 'training/information/best_model.h5'
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    history=model.fit(x_train, y_train, epochs=400, batch_size=64,  validation_data=(x_test, y_test),callbacks=[checkpoint])
    return history

def plot_history(history, model, x_test, y_test):
    accuracy_ontest_data =  model.evaluate(x_test,y_test)[1]*100 + "%"

    epochs = [i for i in range(400)]
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    test_acc = history.history['val_accuracy']
    test_loss = history.history['val_loss']

    fig.set_size_inches(20,6)
    ax[0].plot(epochs , train_loss , label = 'Training Loss')
    ax[0].plot(epochs , test_loss , label = 'Testing Loss')
    ax[0].set_title('Training & Testing Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

    ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
    ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
    ax[1].set_title('Training & Testing Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")

    # Convert plot to PNG image
    png_image = io.BytesIO()
    plt.savefig(png_image, format='png')

    # Encode PNG image to base64 string
    png_image_b64_string = "data:image/png;base64,"
    png_image_b64_string += urllib.parse.quote(base64.b64encode(png_image.getvalue()))
    
    plt.close(fig)  # close the figure

    return png_image_b64_string, accuracy_ontest_data

def decode_predictions(predictions, reverse_mapping):
    decode=[]
    for predict in predictions:
        predict = np.reshape(predict, (1, 8))
        decoded_predictions = [reverse_mapping[np.argmax(predict)] for prediction in predict]
        decode.append(*decoded_predictions)
    
    return decode

def confusion_matrix_plot(actual, predicted, output_file):
    columns = ['calm', 'disgust', 'angry', 'sad', 'neutral', 'fear', 'happy', 'surprise']
    cm = confusion_matrix(actual, predicted)
    df_cm = pd.DataFrame(cm, index=columns, columns=columns)
    fig = plt.figure(figsize=(12,10))
    sns.heatmap(df_cm, annot=True)
    plt.savefig(output_file)
    plt.close(fig)

def evaluate_model(model, x_train, y_train, x_test, y_test, reverse_mapping):
    train_predictions=model.predict(x_train)
    y_train_decoded = decode_predictions(y_train, reverse_mapping)
    train_predictions_decoded = decode_predictions(train_predictions, reverse_mapping)

    print("Training Data Evaluation:")
    print(classification_report(y_train_decoded, train_predictions_decoded))
    confusion_matrix_plot(y_train_decoded, train_predictions_decoded, 'static/images/training_confusion_matrix.png')

    test_predictions=model.predict(x_test)
    y_test_decoded = decode_predictions(y_test, reverse_mapping)
    test_predictions_decoded = decode_predictions(test_predictions, reverse_mapping)

    print("Test Data Evaluation:")
    print(classification_report(y_test_decoded, test_predictions_decoded))
    confusion_matrix_plot(y_test_decoded, test_predictions_decoded, 'static/images/test_confusion_matrix.png')

def main():
    X, Y = load_data()
    X = select_features(X, Y)
    x_train, x_test, y_train, y_test = split_data(X, Y)
    y_train, y_test, label_encoder, onehot_encoder = encode_labels(y_train, y_test)
    x_train, x_test = expand_dims(x_train, x_test)
    model = define_model()
    history = train_model(model, x_train, y_train, x_test, y_test)
    plot_img_b64_str = plot_history(history, model, x_test, y_test)

  #  return plot_img_b64_str

def model_evaluation():
    X, Y = load_data()
    X = select_features(X, Y)
    x_train, x_test, y_train, y_test = split_data(X, Y)
    model =load_model('training/information/best_model.h5')
    with open('training/information/reverse_mapping.pkl', 'rb') as f:
        reverse_mapping = pickle.load(f)
    evaluation_results = evaluate_model(model, x_train, y_train, x_test, y_test, reverse_mapping)
    return evaluation_results
    
    
   



if __name__ == '__main__':
    main()