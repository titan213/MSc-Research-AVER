import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
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
    df = df.groupby('class').apply(lambda x: x.sample(n=df['class'].value_counts().min())).reset_index(drop=True)
    return df['phrase'], df['class']

def vectorize_data(X):
    vectorizer  = TfidfVectorizer(stop_words='english',ngram_range=(1, 2),max_df=0.9,min_df=5)
    return vectorizer.fit_transform(X), vectorizer

def train_model(X, y):
    clf = SVC()

    # hyperparameters to tune
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'degree': [2, 3, 4], # for 'poly' kernel
        'gamma': ['scale', 'auto'], # for 'rbf', 'poly' and 'sigmoid'
    }

    # Create the grid search object
    grid_search = GridSearchCV(clf, param_grid, cv=4)

    # Perform grid search to find the best hyperparameters
    grid_search.fit(X, y)

    return grid_search

def save_model(model, file_path,vectorizer):
    joblib.dump((model, vectorizer),file_path)


def plot_class_distribution(y, path):
    plt.figure(figsize=(10,6))
    y.value_counts().plot(kind='bar')
    plt.xticks(ticks=range(len(y.unique())), labels=y.unique(), rotation=45) 
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.title('Frequency of Each Class')
    plt.tight_layout() 
    plt.savefig(path)
    plt.close()

def plot_confusion_mat(model, X, y, path):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=y.unique())
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues,ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_learning_curve(model, X, y, path):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=4)
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


def train_model_route():
    # Load the data
    X, y = load_data("training/information/sentence.csv")

    # Vectorize the text data
    X_transformed, vectorizer = vectorize_data(X)

    # Train the model
    model = train_model(X_transformed, y)

    # Save the model
    save_model(model,'training/information/svm_model.pkl',vectorizer=vectorizer)
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



