from flask import Flask , render_template , redirect, url_for , request
import numpy as np
import joblib

app = Flask(__name__)

casos = ""

@app.route('/')
def home():
    return render_template('home.html', c = casos)

@app.route('/sobre-proyecto') 
def about():
    return render_template('about.html')

@app.route('/integrantes') 
def integrantes():
    return render_template('integrantes.html')

@app.route('/prediccion' , methods = ["POST"])
def prediccion():
    if(request.method == 'POST'):
        new_deaths_smoothed = request.form["new_deaths_smoothed"]
        new_tests_smoothed = request.form["new_tests_smoothed"] 
        positive_rate = request.form["positive_rate"]
        people_fully_vaccinated = request.form["people_fully_vaccinated"]
        people_vaccinated = request.form["people_vaccinated"]
        X_new = np.array([[new_deaths_smoothed,new_tests_smoothed,positive_rate,people_fully_vaccinated, people_vaccinated]]).astype('float64')
        print(X_new)
        clf = joblib.load('prediction-COVID.pkl')
        pr = clf.predict(X_new)
        print("Predicción : {:.0f} casos".format(np.round(pr[0])) )
        global casos
        casos = "{:.0f}".format(np.round(pr[0])) 
        print(casos)
        return redirect(url_for('home'))
    return render_template('home.html')    




if __name__ == '__main__':
    app.run(debug=True)
    
    