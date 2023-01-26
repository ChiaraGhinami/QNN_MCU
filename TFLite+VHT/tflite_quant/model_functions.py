from keras.models import Sequential
from keras.layers import Dense
from keras import Input, callbacks
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def setup_model():
    model = Sequential()
    model.add(Input(shape=(8,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation="softmax"))
    return model


def nn_model_fit(md, x_tr, y_tr, epochs):

    # batch_size = 50  # Trade-off bw number of iterations to train a NN and the memory required to store input samples

    # To mitigate overfitting the model should be trained for an optimal number of epochs, if no improvement are seen,
    # stop the training
    early_stop = callbacks.EarlyStopping(monitor="val_loss", mode="min", min_delta=0.009, patience=3, restore_best_weights=True)
    hist = md.fit(x_tr, y_tr, epochs=epochs, validation_split=0.2, callbacks=[early_stop])
    return hist


# creating a function for plotting
def plot_epochs(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    epochs_ = range(1, len(loss) + 1)

    plt.plot(epochs_, loss, 'bo', label='Training loss')
    plt.plot(epochs_, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    epochs_ = range(1, len(acc) + 1)

    plt.plot(epochs_, acc, 'bo', label='Training acc')
    plt.plot(epochs_, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.show()


# Compare accuracy of quantized and not quantized model
def keras_models_evaluation(model1, model2, xtest, ytest):
    _, baseline_model_accuracy = model1.evaluate(xtest, ytest, verbose=0)

    _, q_aware_model_accuracy = model2.evaluate(xtest, ytest, verbose=0)

    print('Baseline test accuracy:', baseline_model_accuracy)  # 0.692
    print('Quant test accuracy:', q_aware_model_accuracy)  # 0.690


# This code is needed since tensorflow lite doesn't provide a method for evaluation if the model is not created with
# Model maker, but with a converter, like in this case
def categorical_acc(prediction_digits, test_labels):
    # Compare prediction results with ground truth labels to calculate accuracy.
    accurate_count = 0
    tot_predictions = prediction_digits.shape[0]
    for row_index in range(tot_predictions):
        if np.argmax(prediction_digits[row_index, :]) == np.argmax(test_labels[row_index, :]):
            accurate_count += 1

    accuracy = accurate_count * 1.0 / tot_predictions
    return accuracy


def input_scaling(input_details, features, input_type):
    input_scale, input_zero_point = input_details[0]['quantization_parameters']
    features = (features / input_scale) + input_zero_point
    features = np.around(features)
    # Convert features to NumPy array of expected type
    features = features.astype(input_type)
    return features


def output_scaling(output, output_details):
    output_scale, output_zero_point = output_details[0]['quantization']
    output = output_scale * (output.astype(np.float32) - output_zero_point)
    return output


def layer_details_print(interp_):
    layer_details = interp_.get_tensor_details()
    for layer in layer_details:
        print("\nLayer Name: {}".format(layer['name']))
        print("\tIndex: {}".format(layer['index']))
        print("\n\tShape: {}".format(layer['shape']))
        print("\tTensor: {}".format(interp_.get_tensor(layer['index']).shape))
        print("\tTensor Type: {}".format(interp_.get_tensor(layer['index']).dtype))


"""
Steps for running inference:
    1) Initialize the interpreter and load the interpreter with the Model
    2) Allocate the tensor and get the input and output tensors
    3) Preprocess the input data (in out case only expand dimensions)
    4) Set the processed data as input tensor
    5) Make the inference on the input tensor using the interpreter by invoking it
    6) Obtain the result by mapping the result from the inference
"""


def run_model(model_path_, xtest):
    # Initialize the interpreter
    output_matrix = np.empty((1, 8))
    interpreter = tf.lite.Interpreter(model_path=model_path_)
    interpreter.allocate_tensors()

    # Interpreter.get_input_details()[0] == interpreter.get_input_details()
    # because we only have one input (one dictionary) into the input details array
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # layer_details_print(interpreter)

    # Convert features to NumPy array
    np_features = np.array(xtest)

    # If the expected input type is int8 rescale the input (quantized model, but not this case, our model expects f32)
    input_type = input_details[0]['dtype']
    if input_type == np.int8:
        np_features = input_scaling(input_details, np_features, input_type)

    for i in range(np_features.shape[0]):
        # Add dimension to input sample (TFLite model expects (# samples, data), in this case (1, 8))
        input_data = np.expand_dims(np_features[i], axis=0)
        # Create input tensor out of raw features
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # output_details[0]['index'] = the index which provides the output
        output = interpreter.get_tensor(output_details[0]['index'])
        # If the output type is int8 rescale data
        output_type = output_details[0]['dtype']
        if output_type == np.int8:
            output = output_scaling(output, output_details)
        if i == 0:
            output_matrix = output
        else:  # Add the predicted array to the output matrix
            output_matrix = np.vstack([output_matrix, output])

    return output_matrix
