import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pandas as pd
import numpy as np
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import  OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, regularizers,models
import tensorflow as tf
import sklearn.metrics 
import base64
from io import BytesIO
import matplotlib.pyplot as plt

import seaborn as sns

def load_data(path):
    features = pd.read_csv(path)
    return features

def data_split(features):
    X = features.iloc[: ,:-1].values
    Y = features['labels'].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3 ,random_state=42, shuffle=True)
    return x_train, x_test, y_train, y_test

def data_encode(y_train, y_test):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_train_encoded = y_train_encoded.reshape(-1, 1)

    onehot_encoder = OneHotEncoder()
    y_train_onehot = onehot_encoder.fit_transform(y_train_encoded).toarray()

    y_test_encoded = label_encoder.transform(y_test)
    y_test_encoded = y_test_encoded.reshape(-1, 1)
    y_test_onehot = onehot_encoder.transform(y_test_encoded).toarray()

    reverse_mapping = {encoded_label: original_label for encoded_label, original_label in zip(y_train_encoded.flatten(), y_train)}
    with open('reverse_mapping_plan.pkl', 'wb') as f:
        pickle.dump(reverse_mapping, f)

    y_train = y_train_onehot
    y_test = y_test_onehot

    return y_train, y_test, reverse_mapping

def model_train(train_features, train_labels, val_features, val_labels):
    model = tf.keras.Sequential([
        layers.Conv1D(256, 3, padding='same', activation='relu', input_shape=(162, 1)),
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
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(8, activation='softmax')
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    checkpoint_filepath = 'best_model_test.h5'
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', save_best_only=True, mode='max')

    history = model.fit(train_features, train_labels, epochs=600, batch_size=64, validation_data=(val_features, val_labels), callbacks=[checkpoint])
    model.save('best_test.h5')

    return model, history

def model_evaluate(model, x_test, y_test):
    model = models.load_model('best_test.h5')

    res = model.evaluate(x_test, y_test, batch_size=64)
    for num in range(0, len(model.metrics_names)):
        print(model.metrics_names[num], res[num])

    predicted_probabilities = model.predict(x_test)
    predicted_classes = np.argmax(predicted_probabilities, axis=1)
    print(pd.Series(predicted_classes).value_counts())

    return predicted_probabilities, predicted_classes



def evaluate_classes(predicted_probabilities, predicted_classes, y_test):
    true_classes = np.argmax(y_test, axis=1)

    emotion_mapping = {3: 'fear', 4: 'happy', 7: 'surprise', 2: 'disgust', 6: 'sad', 0: 'angry', 5: 'neutral', 1: 'calm'}
    results = []

    fig, ax = plt.subplots()
    for emotion_code, emotion_name in emotion_mapping.items():
        true_labels_for_emotion = (true_classes == emotion_code)
        predicted_labels_for_emotion = (predicted_classes == emotion_code)

        accuracy = sklearn.metrics.accuracy_score(y_true=true_labels_for_emotion, y_pred=predicted_labels_for_emotion)
        precision = sklearn.metrics.precision_score(y_true=true_labels_for_emotion, y_pred=predicted_labels_for_emotion)
        recall = sklearn.metrics.recall_score(y_true=true_labels_for_emotion, y_pred=predicted_labels_for_emotion)
        f1_score = sklearn.metrics.f1_score(y_true=true_labels_for_emotion, y_pred=predicted_labels_for_emotion)
        auc = sklearn.metrics.roc_auc_score(y_true=true_labels_for_emotion, y_score=predicted_probabilities[:, emotion_code])

        fig_cm, ax_cm = plt.subplots()
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true=true_labels_for_emotion, y_pred=predicted_labels_for_emotion)
        sns.heatmap(confusion_matrix, annot=True, fmt='', cmap='Blues', ax=ax_cm)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for ' + emotion_name)
        plt.xticks([0.5,1.5],['Non-Emotion','Emotion'])
        plt.yticks([0.5,1.5],['Non-Emotion','Emotion'])
        plt.tight_layout()
        buf_cm = BytesIO()
        plt.savefig(buf_cm, format='png')
        plt.close(fig_cm)
        confusion_matrix_img = base64.b64encode(buf_cm.getvalue()).decode('utf-8')

        false_positive_rate, true_positive_rate, _ = sklearn.metrics.roc_curve(y_true=true_labels_for_emotion, y_score=predicted_probabilities[:, emotion_code])
        ax.plot(false_positive_rate, true_positive_rate, label=emotion_name)

        results.append({
            'emotion': emotion_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'auc': auc,
            'confusion_matrix_img': confusion_matrix_img,
        })

    ax.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for all emotions')
    plt.legend(loc='best')
    plt.tight_layout()
    buf_roc = BytesIO()
    plt.savefig(buf_roc, format='png')
    plt.close(fig)
    roc_curve_img = base64.b64encode(buf_roc.getvalue()).decode('utf-8')

    results.append({
        'emotion': 'all_emotions',
        'roc_curve_img': roc_curve_img
    })

    return results


def plot_model_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(15,5)) 

    # summarize history for accuracy
    axs[0].plot(history.history['accuracy']) 
    axs[0].plot(history.history['val_accuracy']) 
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy') 
    axs[0].set_xlabel('Epoch') 
    axs[0].legend(['train', 'validate'], loc='upper left')

    # summarize history for loss
    axs[1].plot(history.history['loss']) 
    axs[1].plot(history.history['val_loss']) 
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss') 
    axs[1].set_xlabel('Epoch') 
    axs[1].legend(['train', 'validate'], loc='upper left')

    # Save it to a bytes buffer.
    bytes_image = BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return base64.b64encode(bytes_image.read()).decode()


def main():
    path = 'training/information/features.csv'  
    features = load_data(path)
    x_train, x_test, y_train, y_test = data_split(features)
    y_train, y_test, reverse_mapping = data_encode(y_train, y_test)
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    model, history = model_train(x_train, y_train, x_test, y_test)
    predicted_probabilities, predicted_classes = model_evaluate(model, x_test, y_test)
    results = evaluate_classes(predicted_probabilities, predicted_classes, y_test)
    model_history = plot_model_history(history)

    return results,model_history






