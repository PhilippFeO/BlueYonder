"""
Everything related to the network itself: Training, test and validation set, architecture, training of the network, plots.
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import LoadData

# Data: 17380 rows (one header, effectively 17379); 17 columns, last one is `cnt`


if __name__ == "__main__":
    data = LoadData.load_data("./Bike-Sharing-Dataset/hour.csv")
    input_hour, label_hour = LoadData.split_data(data)

    # Key values to configure some network parameters
    # set sizes
    test_size = 0.2  # Percentage of test set size
    val_size = 0.3  # Percentage of validation set size

    # Network hyperparameters
    units_layer1 = 32
    units_layer2 = 16
    epochs = 100
    batch_size = 200

    # Fix "randomness" for reproducibility
    np.random.seed(42)

    # Shuffle the data and targets in unison
    indices = np.random.permutation(len(input_hour))
    input_hour = input_hour[indices]
    label_hour = label_hour[indices]

    # Calculate the number of samples for test & validation set
    nmb_test_samples = int(len(input_hour) * test_size)
    nmb_val_samples = int(
        len(input_hour) * (1 - test_size) * val_size)

    # Split the data and targets into training and test sets
    xval = input_hour[:nmb_val_samples]
    yval = label_hour[:nmb_val_samples]
    xtrain = input_hour[nmb_val_samples:-nmb_test_samples]
    ytrain = label_hour[nmb_val_samples:-nmb_test_samples]
    xtest = input_hour[-nmb_test_samples:]
    ytest = label_hour[-nmb_test_samples:]

    # Build model
    model = keras.Sequential([
        layers.Dense(units_layer1, activation='relu'),
        layers.Dense(units_layer2, activation='relu'),
        layers.Dense(1)  # No activation/linear activation function
    ])

    model.compile(optimizer="rmsprop",
                  loss="mse",
                  metrics=["mae"])

    # Train
    history = model.fit(xtrain, ytrain,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(xval, yval))

    # Print mean average deviation
    _, metrics = model.evaluate(xtest, ytest)
    print(f"MAE/MAD of the test set:\n\t{metrics:.3f}")
    print(f"Mean of the test set:\n\t{np.mean(ytest)}")

    # Plots
    # Exclude first point because it is huge and distorts the plot
    accuracy = history.history["mae"][1:]
    val_accuracy = history.history["val_mae"][1:]
    loss = history.history["loss"][1:]
    val_loss = history.history["val_loss"][1:]
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    # plt.show()
