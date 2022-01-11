# Background and Objective

The purpose of this project is to train a model using Keras Tuner, deploy the model on a flask server hosted on an AWS EC2 instance, and create a front end program to query prediction from model.

### Keras Tuner

Keras Tuner allows for automatic grid search of hyperparameters. By instantiating a *units* variable, a Sequential model will be trained for the specified range of parameter values. Keras Tuner can also be used to identify the best optimizer, learning rate, and loss. 

```ruby
def build_model(hp):
    units = hp.Int(name="units", min_value=16, max_value=64, step=16)
    model = keras.Sequential([
        layers.Dense(units, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    optimizer = hp.Choice(name="optimizer", values=["rmsprop", "adam"])
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    return model
```

Rather that calling *fit* on a model, a Keras Tuner object is created, which specifies number of trials, a checkpoint directory, and an overwrite parameter whichs saves the best performing model.

```ruby
tuner = kt.BayesianOptimization(build_model, objective="val_accuracy", max_trials=1, executions_per_trial=2, directory="mnist_kt_test", overwrite=True)
tuner.search(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val), callbacks=callbacks, verbose=2)
```

Remaining code returns best parameters found, creates new model using those parameters, and trains to slightly more than best performing epoch (this new model can be used on training set *combined* with validation set since there is no additional tuning required.)

```ruby
def get_best_epoch(hp):
    model = build_model(hp)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=128, callbacks=callbacks)
    val_loss_per_epoch = history.history["val_loss"]
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
    print(f"Best epoch: {best_epoch}")
    return best_epoch, model

def get_best_trained_model(hp):
    best_epoch, model = get_best_epoch(hp)
    model.fit(x_train_full, y_train_full, batch_size=128, epochs=int(best_epoch * 1.2))
    return model

best_models = []
for hp in best_hps:
    model = get_best_trained_model(hp)
    model.evaluate(x_test, y_test)
    best_models.append(model)
```

The final model state has a test accuracy of 97.10%.

OUTPUT:
```ruby
Epoch 13/100
...
Epoch 18/19
469/469 [==============================] - 9s 19ms/step - loss: 0.0178 - accuracy: 0.9958
Epoch 19/19
469/469 [==============================] - 9s 19ms/step - loss: 0.0171 - accuracy: 0.9962
313/313 [==============================] - 1s 3ms/step - loss: 0.1352 - accuracy: 0.9710
```

### Running Dockerfile on EC2 Instance

To deploy model on EC2 instance, a Flask app was created which, upon startup, would load model and listen for requests. A Dockerfile was used to automatically update virtual machine, install requirements, and launch the python file. 

```ruby
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
```

The following steps were followed to deploy the model to production ready instance :
1. Launch Ubuntu Amazon Machine Image and set up security groups to allow traffic on specified port
2. SSH into instance using PuTTY, public DNS, and access key
3. Update linux system and install Docker
4. Run following command which will bind Virtual Machine port 5000 to the Flask server running on port 5000 (pulls Dockerfile from DockerHub)

```
sudo docker run -p 5000:5000/tcp {DockerHub profile}/{DockerHub repo}
```

### Sample End User Program

In conclusion, this process could be followed to launch any kind of model on a server at production level. Additional functionality could be written into the flask app to handle various requests, and this would have use cases similar to web based translation or photo tagging. Additional security protocols should be implemented. 

```ruby
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
```
Example output :

This number is a 6

![](example_image?raw=true)