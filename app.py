from flask import Flask, request, render_template, redirect, url_for, jsonify,send_from_directory,session
from werkzeug.utils import secure_filename
import os
import predict_module
import matplotlib.pyplot as plt
import librosa.display
import os
import information_classification
import train_voice_emotion_model as tvem
from datetime import datetime
import shutil 
import csv
import zipfile
import extract_voice_features as evf


#global variables
filepath = ""
emotion = ""
department = ""
sentence = ""

def create_waveplot(data, sr, filename):
    if not os.path.exists('static'):
        os.makedirs('static')
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio' , size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.savefig(filename) # save figure
    plt.close() # close figure

def create_spectrogram(data, sr, filename):
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio', size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.savefig(filename) # save figure
    plt.close() # close figure

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.secret_key = "testing_secret"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/statics/uploads', methods=['GET', 'POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file in request")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        print("File name is empty")
        return redirect(request.url)

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    button = request.form.get('button')
    if button == 'submit':
        data, sr = librosa.load(filepath, sr=None)
        waveplot_filename = 'static/waveplot.png'
        spectrogram_filename = 'static/spectrogram.png'
        create_waveplot(data, sr, waveplot_filename)
        create_spectrogram(data, sr, spectrogram_filename)
        audio_file= 'static/uploads/' + filename
        emotion, department, sinhala_sentence, sentence = predict_module.predict(filepath)


        print('Data retrieved from session:', session)

        session['filepath'] = filepath
        session['emotion'] = emotion
        session['department'] = department
        session['sentence'] = sentence

        return render_template('index.html', 
                            waveplot=waveplot_filename, 
                            spectrogram=spectrogram_filename,
                            emotion=emotion,
                            department=department,
                            audio=audio_file,
                            sentence=sentence,
                            sinhala_sentence=sinhala_sentence)
    
    
@app.route('/resolve_incident', methods=['GET', 'POST'])
def resolve_incident():
    button = request.form.get('button')
    if button == 'resolveIncident':
        filepath = session.get('filepath')
        emotion = session.get('emotion')
        department = session.get('department')
        sentence = session.get('sentence')

        if filepath and emotion and department and sentence:
            new_filepath = os.path.join('training/voice_data/local', 
                                        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{emotion}.wav')
            shutil.move(filepath, new_filepath)
            filepath = new_filepath

            # Calculate the new id for the sentence
            with open('training/information/sentence.csv', 'r') as f:
                last_line = f.readlines()[-1]
                last_id = int(last_line.split(",")[0])
                new_id =  last_id + 1
                
            # Append the new sentence and department to the CSV
            sentence = sentence.replace(',', ' ')
            with open('training/information/sentence.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow('\n')
                writer.writerow([new_id, sentence, department])  

    return render_template('index.html')

    
#text classification model and page
@app.route('/train_classification_model', methods=['GET', 'POST'])
def train_classification_model_page():
    if request.method == 'POST':
        # Train the model here
        hyperparameters, best_Accuracy = information_classification.train_model_route()

        # Then return the template with the results
        return render_template('train_classification_model.html',
                               hyperparameters=hyperparameters,
                               best_Accuracy=best_Accuracy)
    else:
        # If it's not a POST request, render the page normally
        return render_template('train_classification_model.html')
    
#apply new classification moldel
@app.route('/apply_new_text_classification', methods=['GET', 'POST'])
def apply_new_text_classification_model():

    button = request.form.get('button')
    src_folder = "training/information"
    dst_folder = "models"
    backup_folder = "backups/text_classification"
    backupFileName = f"{backup_folder}/{time}.zip"
    filenames = ['sentence.csv', 'svm_model.pkl']
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if button == 'applyNewTextClassification':

        #backing up the old model
        with zipfile.ZipFile(backupFileName, 'w') as zipObj:
            for filename in filenames:
                source = os.path.join(dst_folder, filename)
                zipObj.write(source, arcname=filename)   

        for filename in filenames:
            source = os.path.join(src_folder, filename)
            destination = os.path.join(dst_folder, filename)
            shutil.copy2(source, destination)

        return render_template('train_classification_model.html')

@app.route('/train_emotion_model', methods=['GET', 'POST'])
def train_emotion_model_page():
    if request.method == 'POST':
        evf.main()
        metrics,model_history  =  tvem.main()
        return render_template('train_emotion_model.html',metrics=metrics,model_history=model_history)
    else:
        return render_template('train_emotion_model.html')

@app.route('/apply_new_emotion_model', methods=['GET', 'POST'])
def apply_new_emotion_model():
    button = request.form.get('button')
    src_folder = "training/voice_data"
    dst_folder = "models"
    backup_folder = "backups/emotion"
    backupFileName = f"{backup_folder}/{time}.zip"
    filenames = ['reverse_mapping.pkl','best_model.h5']
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if button == 'applyNewEmotionModel':

        #backing up the old model
        with zipfile.ZipFile(backupFileName, 'w') as zipObj:
            for filename in filenames:
                source = os.path.join(dst_folder, filename)
                zipObj.write(source, arcname=filename)   

        for filename in filenames:
            source = os.path.join(src_folder, filename)
            destination = os.path.join(dst_folder, filename)
            shutil.copy2(source, destination)

        return render_template('train_emotion_model.html')




if __name__ == '__main__':
    app.run(debug=True)
