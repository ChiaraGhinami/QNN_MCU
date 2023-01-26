import numpy as np
from model_functions import *
import pandas as pd
from sklearn.model_selection import train_test_split
from qkeras import *
import keras.backend as K
from qkeras.utils import quantized_model_debug
from keras.optimizers import Adam


def import_csv_file(data_dir):
    df = pd.read_csv(data_dir)
    return df


def print_df_info(df):
    print(df.head())
    print(df.shape)
    print("Value Count :\n", df["class"].value_counts())


def data_norm(x_tr, x_ts):
    mean_tr = x_tr.mean(axis=0)
    std_tr = x_tr.std(axis=0)
    mean_ts = x_ts.mean(axis=0)
    std_ts = x_ts.std(axis=0)

    x_tr -= mean_tr
    x_tr /= (2*std_tr)
    x_tr = x_tr.clip(-1, 1)

    x_ts -= mean_ts
    x_ts /= (2*std_ts)
    x_ts = x_ts.clip(-1, 1)
    return x_tr, x_ts


# OneHot encoding of test and training labels, it must always be done with categorical_crossentropy loss
def onehot_enc(y_tr, y_ts):
    y_tr = tf.keras.utils.to_categorical(y_tr)
    y_ts = tf.keras.utils.to_categorical(y_ts)
    return y_tr, y_ts


def train_test_splitting(df):
    # Select columns with 'float64' dtype
    float64_cols = list(df.select_dtypes(include='float64'))
    # Set the type of these columns to float32
    df[float64_cols] = df[float64_cols].astype('float32')
    # Get the features array & the class (labels) array
    features = df.drop(columns=["label", "class", "time"]).values
    classes = df["class"].values
    x_tr, x_ts, y_tr, y_ts = train_test_split(features, classes, test_size=0.2, random_state=1)
    x_tr, x_ts = data_norm(x_tr, x_ts)
    return x_tr, x_ts, y_tr, y_ts


# Save x_test and y_test on a file for CMSIS inference
def xy_test_save(x_ts, y_ts, n):
    f = open("C:/Users/chiar/OneDrive/Desktop/x_test1.txt", "w+")
    f.write(str(list(np.ravel(x_ts[n:n+100, :]))))
    f.close()

    f1 = open("C:/Users/chiar/OneDrive/Desktop/y_test.txt", "w+")
    yts_idx = [np.argmax(y_ts[i+n, :]).astype('int') for i in range(100)]
    f1.write(str(yts_idx))
    #f1.write(str(list(np.ravel(y_ts[:n, :].astype('int')))))
    f1.close()


# This function quantizes the parameters, reshapes them in a CMSIS compliant format and saves them on a file
# After the QAT parameters are not quantized, they're just trained in a way that minimizes the quantization error
# This function was copied by a bigger QKeras function model_save_quantized_weights() taken by
# https://github.com/google/qkeras/blob/master/qkeras/utils.py I tried to use the function but it wasn't working
def save_parameters(model, mul_):
    par_list = []
    file = open("C:/Users/chiar/OneDrive/Desktop/q8_par.txt", "w+")
    for layer in model.layers:
        if hasattr(layer, "get_quantizers"):  # Some layers don't have quantizer (ex the last softmax)
            qs = layer.get_quantizers()
            ws = layer.get_weights()
            for quantizer, weight in zip(qs, ws):
                if quantizer:
                    weight = tf.constant(weight)
                    weight = tf.keras.backend.eval(quantizer(weight))
                    weight = np.array(weight)
                    hw_weight = np.round(weight * mul_)
                    if hw_weight.ndim > 1:  # If hw_weight are weights and not bias, reshape them
                        hw_weight = reshape_weights(hw_weight)
                    else:
                        hw_weight = np.transpose(hw_weight)
                    par_list.append(hw_weight)
                    file.write("\n\n" + layer.name + " ")
                    hw_weight.tofile(file, sep=", ", format="%d")
    file.close()
    return par_list


def reshape_weights(weights):
    transposed_wts = np.transpose(weights)
    new_weights = convert_weights(
        np.reshape(transposed_wts, (transposed_wts.shape[0], transposed_wts.shape[1], 1, 1)))
    return new_weights


def convert_weights(weights):
    [r, h, w, c] = weights.shape
    weights = np.reshape(weights, (r, h*w*c))
    new_weights = np.copy(weights)
    new_weights = np.reshape(new_weights, (r*h*w*c))
    return new_weights


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # **************Load data**************
    dataset = import_csv_file('C:/Progetto/Quantization/EMG_data.csv')
    # print_df_info(dataset)  # (4237907, 11)

    # **************Prepare data**************
    # Drop unmarked data & obtain train and test set
    index_names = dataset[dataset['class'] == 0].index
    dataset.drop(index_names, inplace=True)
    x_train, x_test, y_train, y_test = train_test_splitting(dataset)
    y_train, y_test = onehot_enc(y_train, y_test)

    # Define number of bits for the quantization. weights have a range between [-2 ; 2] so we cannot
    # multiply by 2**(n_bits-1), ex 128 with int8, otherwise they'll be out of range [-128 ; 127], you can force them
    # to be in the range [-1 ; 1] by changing the number of integer bits to 0, instead of 1
    n_bits = 8
    parameter_mul = 64
    input_mul = 128

    xq_test = np.round(x_test*input_mul)
    # Save in a file the first 100 quantized inputs & the true predictions
    xy_test_save(xq_test, y_test, 100)

    # Create the qkeras model and fit it
    qmodel = create_qmodel(n_bits)
    qmodel.compile(optimizer=Adam(0.0005), loss="categorical_crossentropy", metrics=["accuracy"])
    qmodel.summary()
    history = nn_model_fit(qmodel, x_train, y_train, 30)

    # Measure the model accuracy on the test set
    _, qmodel_accuracy = qmodel.evaluate(x_test, y_test, verbose=0)
    print('Quant test accuracy:', qmodel_accuracy)   # 0.69
    quantized_model_debug(qmodel, x_test, plot=False)
    # Save weights and biases on a file for CMSIS inference
    saved_par = save_parameters(qmodel, parameter_mul)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
