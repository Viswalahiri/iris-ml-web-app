import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('../models/model.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    if(prediction[0]==0):
        output = "Setosa"
    elif(prediction[0]==1):
        output = "Versicolor"
    else:
        output = "Virginica"

    return render_template('index.html', prediction_text='The species of Iris flower is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
