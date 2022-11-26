import numpy as np
from flask import Flask, request, render_template
import joblib 

app = Flask(__name__)
model = joblib.load(open('rf_hypertuned_model_new.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output ==1:
         output = "Maaf, Anda kemungkinan terkena penyakit jantung"
    else : 
        output ="Anda tidak memiliki kemungkinan terkena penyakit jantung"

    return render_template('index.html', prediction_text='Kemungkinan terkena penyakit jantung : {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)