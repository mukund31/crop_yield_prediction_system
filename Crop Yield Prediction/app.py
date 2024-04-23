from flask import Flask,request, render_template
import numpy as np
import pickle
import sklearn
print(sklearn.__version__)

dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        year = request.form['Year']
        av_rain = request.form['average_rain_fall_mm_per_year']
        pesticides = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        area = request.form['Area']
        item  = request.form['Item']

        features = np.array([[year,av_rain,pesticides,avg_temp,area,item]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1,-1)

        # return render_template('index.html',prediction = prediction)
        prediction = dtr.predict(transformed_features).reshape(1,)

        return render_template('index.html', prediction=prediction[0])


if __name__=="__main__":
    app.run(debug=True)