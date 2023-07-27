import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV,train_test_split
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')


def load_data(file_path):
    df = pd.read_csv(file_path)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df['phrase'], df['class']

def vectorize_data(X):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(X), vectorizer

def train_model(X, y):
    # Create the SVM classifier
    clf = SVC()

    # Define the hyperparameters to tune
    param_grid = {
        'C': [1, 10, 100],
        'kernel': ['linear', 'rbf'],
    }

    # Create the grid search object
    grid_search = GridSearchCV(clf, param_grid, cv=5)

    # Perform grid search to find the best hyperparameters
    grid_search.fit(X, y)

    return grid_search

def save_model(model, file_path):
    joblib.dump(model, file_path)

def classify_sentence(model, vectorizer, sentence):
    # Convert the sentence to a feature vector
    sentence_transformed = vectorizer.transform([sentence])

    # Predict the label of the sentence
    return model.predict(sentence_transformed)

def plot_class_distribution(y, path):
    plt.figure(figsize=(10,6))
    y.value_counts().plot(kind='bar')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.title('Frequency of Each Class')
    plt.savefig(path)
    plt.close()

def plot_confusion_mat(model, X, y, path):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(path)
    plt.close()

def plot_learning_curve(model, X, y, path):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(path)
    plt.close()


def main():
    # Load the data
    X, y = load_data("training/information/sentence.csv")
    
    # Plot class distribution
    plot_class_distribution(y)

    # Vectorize the text data
    X_transformed, vectorizer = vectorize_data(X)

    # Train the model
    model = train_model(X_transformed, y)

    # Save the model
    save_model(model, 'training/information/svm_model.pkl')

    # Print the best hyperparameters and the corresponding accuracy
    print("Best Hyperparameters:", model.best_params_)
    print("Best Accuracy:", model.best_score_)

    # Plot the learning curve
    plot_learning_curve(model, X_transformed, y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    # Plot confusion matrix
    model.fit(X_train, y_train)  # Fit the model on the training set
    plot_confusion_mat(model, X_test, y_test)

    # Classify a new sentence
    sentence = "I'm having a heart attack."
    label = classify_sentence(model, vectorizer, sentence)
    print(label)

def train_model_route():
    # Load the data
    X, y = load_data("training/information/sentence.csv")

    # Vectorize the text data
    X_transformed, vectorizer = vectorize_data(X)

    # Train the model
    model = train_model(X_transformed, y)

    # Save the model
    save_model(model, 'training/information/svm_model.pkl')
    # Print the best hyperparameters and the corresponding accuracy
    hyperparameters = model.best_params_
    best_Accuracy= model.best_score_

    # Save plots as images
    plot_class_distribution(y, 'static/images/class_distribution.png')
    plot_learning_curve(model, X_transformed, y, 'static/images/learning_curve.png')

     # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    # Fit the model on the training set and plot confusion matrix
    model.fit(X_train, y_train)
    plot_confusion_mat(model, X_test, y_test, 'static/images/confusion_matrix.png')

    return hyperparameters, best_Accuracy



