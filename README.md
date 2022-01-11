# Background and Objective

The purpose of this project is to train a model using Keras Tuner, deploy the model on a flask server hosted on an AWS EC2 instance, and create a front end program to query prediction from model.

### Keras Tuner

Keras Tuner allows for automatic grid search of hyperparameters. By instantiating a *units* variable, a Sequential model will be trained for the specified range of parameter values. Keras Tuner can also be used to identify the best optimizer, learning rate, and loss. Rather that calling *fit* on a model, a Keras Tuner object is created, which specifies number of trials, a checkpoint directory, and an overwrite parameter whichs saves the best performing model.

'''def build_model(hp):
    units = hp.Int(name="units", min_value=16, max_value=64, step=16)
    model = keras.Sequential([
        layers.Dense(units, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    optimizer = hp.Choice(name="optimizer", values=["rmsprop", "adam"])
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    return model

tuner = kt.BayesianOptimization(build_model, objective="val_accuracy", max_trials=1, executions_per_trial=2, directory="mnist_kt_test", overwrite=True)'''