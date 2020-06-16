from flask import Flask
from markupsafe import escape
import joblib
import numpy as np
knn = joblib.load('knn.pkl')  

app = Flask(__name__)

@app.route('/')
def hello_world():
    print(1+2)
    return '<h1>Hello, my best friend (Docker)!</h1>'

@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % escape(username)


@app.route('/iris/<param>')
def iris(param):

    param = param.split(',')
    param = [float(num) for num in param]
     
    
    param = np.array(param).reshape(1, -1)
    predict = knn.predict(param)

    return str(predict)