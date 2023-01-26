from keras.models import Sequential
from keras.layers import Activation
from keras import Input, callbacks
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu


def create_qmodel(n_bits):
    model = Sequential()
    model.add(Input(shape=(8,)))
    model.add(QDense(128, name="fc1", kernel_quantizer=quantized_bits(n_bits, 1, alpha="auto_po2"),
                     bias_quantizer=quantized_bits(n_bits, 1, alpha="auto_po2")))
    model.add(QActivation(activation=quantized_relu(n_bits, 1), name="relu1"))
    model.add(QDense(64, name="fc2", kernel_quantizer=quantized_bits(n_bits, 1, alpha="auto_po2"),
                     bias_quantizer=quantized_bits(n_bits, 1, alpha="auto_po2")))
    model.add(QActivation(activation=quantized_relu(n_bits, 1), name="relu2"))
    model.add(QDense(8, name="fc3", kernel_quantizer=quantized_bits(n_bits, 1, alpha="auto_po2"),
                     bias_quantizer=quantized_bits(n_bits, 1, alpha="auto_po2")))
    model.add(Activation(activation="softmax", name="softmax"))
    return model


def nn_model_fit(md, x_tr, y_tr, epochs):

    # batch_size = 50  # Trade-off bw number of iterations to train a NN and the memory required to store input samples

    # To mitigate overfitting the model should be trained for an optimal number of epochs, if no improvement are seen,
    # stop the training
    early_stop = callbacks.EarlyStopping(monitor="val_loss", mode="min", min_delta=0.009, patience=5, restore_best_weights=True)
    hist = md.fit(x_tr, y_tr, epochs=epochs, validation_split=0.2, callbacks=[early_stop])

    return hist
