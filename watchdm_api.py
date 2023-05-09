import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from flask import Flask, request, url_for, redirect, render_template, jsonify
import pickle
import joblib

#import portpicker
#port = portpicker.pick_unused_port()

#from gevent.pywsgi import WSGIServer
#host='localhost'



app = Flask(__name__)

#rsf = pickle.load(open('C:/Aditya/watch_dm_rsf_model.pkl','rb'))

rsf = joblib.load("RF_compressed.joblib")


cols = ['Age', 'BMI', 'SBP', 'DBP', 'SCreat', 'FPG', 'HbA1c', 'HDL', 'QRS', 'hxMI', 'hxCABG']
##x = [75.052055,	15.0,	152,	63,	0.9,	131.0,	8.070549,	33.0,	0,	1,	1]


@app.route('/predict',methods=['POST'])
def predict():
    jdata = request.json
    jdata = json.dumps(jdata)
    #data_unseen = pd.read_json(jdata, orient = 'records' )
    data_unseen = pd.read_json(jdata)
    prediction = rsf.predict(data_unseen)
    output = prediction.round(decimals= 2)
    return jsonify({"prediction":list(output)})

if __name__ == "__main__":
    app.run(debug=True)


