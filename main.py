from flask import Flask, jsonify, render_template, request
from log import LogClass
from RobertaModel import RobertClass
from Config import ConfigClass

# Initialised the flask app
app = Flask(__name__)

RobertaObj = RobertClass()
ConfigObj = ConfigClass("params.yaml")
configData = ConfigObj.Loading_Config()
loggerObj = LogClass(configData['LoggingFileName'])


@app.route("/", methods=["GET"])
def HomePage():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def PredictionRoute():
    if request.json['text'] is not None:
        try:
            text = request.json['text']
            loggerObj.Logger("Get The Text from USER : " + str(text))
            result_dic = RobertaObj.Prediction(text)
            loggerObj.Logger("Get the result & giving to the USER")
            return jsonify(result_dic)
        except Exception as e:
            return "Exception Occured in PredictionRoute, " + str(e)


if __name__ == "__main__":
    app.run()
