import tensorflow as tf
from tensorflow import keras
import numpy as np
from flask import Flask, request
import ast


model = None
app = Flask(__name__)

def load_model():
    global model
    _, (x_test, _) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    x_test = x_test.reshape((-1, 28 * 28)).astype("float32") / 255
    model = keras.models.load_model("./mnist_model")
    print("loaded model")


@app.route("/")
def home_endpoint():
    return "Hello World!"


@app.route("/predict", methods=["POST"])
def get_prediction():
    if request.method == "POST":
        data = request.get_json()
        data = ast.literal_eval(data)
        data = np.asarray(data["instances"])
        data = np.reshape(data, (1, 784))
        prediction = model.predict(data)
        prediction = str(np.where(prediction[0]==max(prediction[0]))[0][0])
    return "This number is a " + prediction


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)

