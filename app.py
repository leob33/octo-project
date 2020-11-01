from flask import Flask,request, render_template, jsonify
from joblib import load
from function_transformer import *


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

model = load("pipeline_lr.joblib")

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
    app.run(debug=True)
