import requests
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import json
import numpy as np


_, (x_test, _) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
x_test = x_test.reshape((-1, 28 * 28)).astype("float32") / 255
x_test = x_test.tolist()
pred_this = random.choice(x_test)


def rest_request(data):
    payload = json.dumps({"instances": data})
    response = requests.post(url="http://ec2-18-217-188-176.us-east-2.compute.amazonaws.com:5000/predict", json=payload)
    return response


r = rest_request(pred_this)
print(str(r.content).split("'")[1])
pred_this_copy = np.asarray(pred_this).reshape(28, 28, 1)
imgplot = plt.imshow(pred_this_copy)
plt.show()
