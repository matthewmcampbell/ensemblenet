import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

class EnsembleLayer(layers.Layer):

    def __init__(self, dense_units, num_classes):
        super(EnsembleLayer, self).__init__()
        # self.inp = keras.Input()
        self.conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.flat = layers.Flatten()
        self.denses = [layers.Dense(d, activation="relu") for d in dense_units]
        self.dropout = layers.Dropout(0.5)
        self.probs = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        # x = keras.Input(shape = input_shape)(inputs)
        # print(x)
        # x = self.inp(inputs)
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flat(x)
        for dense in self.denses:
            x = dense(x)
        x = self.dropout(x)
        return self.probs(x)

class EnsembleUnit(keras.Model):

    def __init__(self, dense_units, num_classes):
        super(EnsembleUnit, self).__init__()
        self.block = EnsembleLayer(dense_units, num_classes)

    def call(self, inputs):
        return self.block(inputs)

class EnsembleModel(keras.Model):

    def __init__(self, many_dense_units, num_classes, compile_params, fit_params):
        super(EnsembleModel, self).__init__()
        self.many_dense_units = many_dense_units
        self.submodels = [EnsembleUnit(dense_units, num_classes) for dense_units in many_dense_units]
        [model.compile(**compile_params) for model in self.submodels]
        [model.fit(*fit_params[0], **fit_params[1]) for model in self.submodels]
        for model in self.submodels:
            model.trainable = False
        self.Dense1 = layers.Dense(1500, activation = 'relu')
        self.out_layer = layers.Dense(10, activation = "softmax")

    def call(self, inputs):
        probs = [model(inputs) for model in self.submodels]
        prob_layer = tf.concat(probs, axis=1)
        x = self.Dense1(prob_layer)
        return self.out_layer(x)

    def get_model_accuracies(self, x_test, y_test):
        
        for i, model in enumerate(self.submodels):
            m = keras.metrics.CategoricalAccuracy()
            m.update_state(model(x_test), y_test)
            print(self.many_dense_units[i])
            print(m.result().numpy())
