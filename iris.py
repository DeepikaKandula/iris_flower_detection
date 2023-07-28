from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sklearn
import os
import pickle
import warnings

app = Flask(__name__)


dir = "D:\iris"
file_name = "iris.pkl"
filepath = os.path.join(dir, file_name)

with open(filepath, "rb") as file:
    loaded_model = pickle.load(file)


@app.route("/")
def home():
    return render_template("iris.html")


@app.route("/predict", methods=["POST"])
def predict():
    Sepal_Len = float(request.form["Sepal Length"])
    Sepal_Wid = float(request.form["Sepal Width"])
    Petal_Len = float(request.form["Petal Length"])
    Petal_Wid = float(request.form["Sepal Width"])

    feature_list = [Sepal_Len, Sepal_Wid, Petal_Len, Petal_Wid]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = loaded_model.predict(single_pred)

    flower_dict = {1: "setosa", 2: "versicolor", 3: "virginica"}


    return render_template("iris.html",prediction=prediction)

    """if prediction[0] in flower_dict:
        flower = flower_dict[prediction[0]]
    else:
        result = "Sorry,we could the classify the flower using the above data"

    return render_template("iris.html", prediction=result)"""


if __name__ == "__main__":
    app.run(debug=True)
