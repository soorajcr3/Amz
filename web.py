from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)
model_rf=pickle.load(open('Aamzon_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data1 = request.form['amt']
    data2 = request.form['cat']
   
    arr = np.array([[data1,data2]])
    output = model_rf.predict(arr)
    return render_template('result.html', prediction_text="Order quantity is {}".format(output))

@app.route('/categories')
def categories():
    return render_template('categories.html')

if __name__ == '__main__':
    app.run(debug=True)