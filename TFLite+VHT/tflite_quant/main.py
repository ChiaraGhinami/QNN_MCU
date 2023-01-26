import numpy as np
from model_functions import *
import pandas as pd
import tempfile
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.lite.python.util import convert_bytes_to_c_source


def import_csv_file(data_dir):
    df = pd.read_csv(data_dir)
    return df


def print_df_info(df):
    print(df.head())
    print(df.shape)
    print("Value Count :\n", df["class"].value_counts())


def data_norm(x_tr, x_ts):
    mean = x_tr.mean(axis=0)
    std = x_tr.std(axis=0)

    x_tr -= mean
    x_tr /= std

    x_ts -= mean
    x_ts /= std
    return x_tr, x_ts


# OneHot encoding of test and training labels, it must always be done with categorical_crossentropy loss
def onehot_enc(y_tr, y_ts):
    y_tr = tf.keras.utils.to_categorical(y_tr)
    y_ts = tf.keras.utils.to_categorical(y_ts)
    return y_tr, y_ts


def train_test_splitting(df):
    # Select columns with 'float64' dtype
    float64_cols = list(df.select_dtypes(include='float64'))
    # Set the type of theese columns to float32
    df[float64_cols] = df[float64_cols].astype('float32')
    # Get the feature array & the class (labels) array
    features = df.drop(columns=["label", "class", "time"]).values
    classes = df["class"].values
    x_tr, x_ts, y_tr, y_ts = train_test_split(features, classes, test_size=0.2, random_state=1)
    x_tr, x_ts = data_norm(x_tr, x_ts)
    return x_tr, x_ts, y_tr, y_ts


# Save x_test and y_test on a file for keil MDK
def xy_test_save(x_ts, y_ts, n):
    #x_ts = np.round(x_ts[:n, :]*128)
    file = open("C:/Users/chiar/OneDrive/Desktop/x_test.txt", "w+")
    file.write(str(list(np.ravel(x_ts[n:n+100, :]))))
    file.close()

    file1 = open("C:/Users/chiar/OneDrive/Desktop/y_test.txt", "w+")
    file1.write(str(list(np.ravel(y_ts[n:n+100, :].astype('int')))))
    file1.close()


def measure_size(file_, model, text_):

    with open(file_, 'wb') as f:
        f.write(model)

    print(text_, os.path.getsize(file_) / float(2 ** 20))  # 40kB


# Create a source file and a header file
def convert_to_c(model, text_, path):

    source_text, header_text = convert_bytes_to_c_source(model, text_)
    with open(path+'NeuralNetwork.h', 'w') as file:
        file.write(header_text)

    with open(path+'NeuralNetwork.cpp', 'w') as file:
        file.write(source_text)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # ***********************Load data************************
    dataset = import_csv_file('C:/Progetto/Quantization/EMG_data.csv')
    # print_df_info(dataset)  # (4237907, 11)

    # **********************Prepare data**********************
    # Drop unmarked data & obtain train and test set
    index_names = dataset[dataset['class'] == 0].index
    dataset.drop(index_names, inplace=True)
    x_train, x_test, y_train, y_test = train_test_splitting(dataset)
    y_train, y_test = onehot_enc(y_train, y_test)
    #xy_test_save(x_test, y_test, 900)

    # **********************FFNN training***********************
    NN_model = setup_model()
    NN_model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
    NN_model.summary()
    history = nn_model_fit(NN_model, x_train, y_train, 20)
    plot_epochs(history)

    # ********************Model quantization*********************
    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(NN_model)
    # The categorical_crossentropy loss is used in multiclass classification, as this case, accuracy--> categorical
    q_aware_model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
    q_aware_model.summary()
    history = nn_model_fit(q_aware_model, x_train, y_train, 20)

    # Calculate the categorical accuracy for the non-quantized and for the quantized model
    keras_models_evaluation(NN_model, q_aware_model, x_test, y_test)

    # ******************Convert to a TFLite model******************
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()
    
    # Save the model
    tflite_model_path = "C:/Users/chiar/OneDrive/Desktop/UNICA/Tesi/Progetto/Quantization/Code/Python/FFNN_model.tflite"
    open(tflite_model_path, "wb").write(quantized_tflite_model)

    # ******************Measure model sizes*************************

    # Create float TFLite model.
    float_converter = tf.lite.TFLiteConverter.from_keras_model(NN_model)
    float_tflite_model = float_converter.convert()
    _, float_file = tempfile.mkstemp('.tflite')
    measure_size(float_file, float_tflite_model, "Float model in Mb: ")  # 40kB

    _, quant_file = tempfile.mkstemp('.tflite')
    measure_size(quant_file, quantized_tflite_model, "Quantized model in Mb: ")  # 13KB
    
    # ******************Convert code to a c++ source*****************
    dest_path = 'C:/Users/chiar/OneDrive/Desktop/UNICA/Tesi/Progetto/Quantization/'
    convert_to_c(quantized_tflite_model, "FFNN_model", dest_path)  # Don't leave spaces in the string (not "FFNN model")

    # **************************Run inference************************
    tflite_model_path = "C:/Users/chiar/OneDrive/Desktop/UNICA/Tesi/Progetto/Quantization/Code/Python/FFNN_model.tflite"
    out_matrix = run_model(tflite_model_path, x_test)  # This should be done with enough test samples, otherwise the
    acc_ = categorical_acc(out_matrix, y_test)         # accuracy value is not accurate, with 100 samples acc was 80%
    print(acc_)  # 0.6895 ---> tensorflow lite model conversion do not decrease the accuracy


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
