from flask import Flask,request, render_template, jsonify
from joblib import load
import pandas as pd
import numpy as np

from sklearn.preprocessing import FunctionTransformer, LabelBinarizer, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

cols = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose']
datatype = {
    'Age': int,
    'Sex': str,
    'Job': int,
    'Housing': str,
    'Saving accounts': str,
    'Checking account': str,
    'Credit amount': int,
    'Duration': int,
    'Purpose': str}

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict', methods=["POST"])
def predict():
    content = [x for x in request.form.values()]
    data = pd.DataFrame([content], columns=cols)
    data = data.astype(datatype)
    prediction = model.predict(data)[0]
    res = 'high risk' if prediction == 0 else "low risk" if prediction == 1 else "error"
    return render_template('home.html', pred='Our model evaluate your credit risk as {}'.format(res))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data = pd.DataFrame([data])
    data = data.astype(datatype)
    prediction = model.predict(data)[0]
    res = 'high risk' if prediction == 0 else "low risk" if prediction == 1 else "error"
    return jsonify(res)


if __name__ == '__main__':
    def categorize_age(X):
        X['Age_cat'] = pd.cut(X.Age, (18, 25, 35, 60, 120), labels=['Baby', 'Young', 'Adult', 'Senior'])
        return X
    def fill_na(X):
        return X.replace(np.nan, "unknown")
    model = load("pipeline_lr.joblib")
    app.run(debug=True)
