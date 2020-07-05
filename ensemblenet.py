import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

default_conv_features = [16, 32]


class EnsembleLayer(layers.Layer):

    def __init__(self, dense_units, num_classes, conv_features=None):
        super(EnsembleLayer, self).__init__()
        if conv_features is None:
            conv_features = default_conv_features
        assert len(conv_features) == 2
        c1, c2 = conv_features
        self.conv1 = layers.Conv2D(c1, kernel_size=(3, 3), activation="relu")
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = layers.Conv2D(c2, kernel_size=(3, 3), activation="relu")
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.flat = layers.Flatten()
        self.denses = [layers.Dense(d, activation="relu") for d in dense_units]
        self.dropout = layers.Dropout(0.5)
        self.probs = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
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

    def __init__(self, dense_units, num_classes, conv_features=None):
        super(EnsembleUnit, self).__init__()
        self.block = EnsembleLayer(dense_units, num_classes, conv_features)

    def call(self, inputs):
        return self.block(inputs)


class EnsembleModel(keras.Model):

    def __init__(self, many_dense_units, num_classes, compile_params,
                 fit_params, many_conv_features=None, trainable=False,
                 logging=None):

        def update_log_callback(logging, fit_params):
            tensorboard_callback_sub = tf.keras.callbacks.TensorBoard(
                log_dir=logging, histogram_freq=1)
            fit_params[1]['callbacks'] = [tensorboard_callback_sub]
            return None

        def print_structure(i):
            print('Structure of submodel {}:'.format(i + 1))
            print('\tConv shapes: {}'.format(many_conv_features[i])) if not (
                    many_conv_features[i] is None) else print(
                '\tConv shapes: {}'.format(default_conv_features))
            print('\tDense shapes: {}'.format(many_dense_units[i]))
            return None

        super(EnsembleModel, self).__init__()
        if many_conv_features is None:
            many_conv_features = [None] * len(many_dense_units)
        assert len(many_dense_units) == len(many_conv_features)
        self.many_dense_units = many_dense_units
        self.num_classes = num_classes
        self.submodels = [EnsembleUnit(dense_units, num_classes, conv_features)
                          for dense_units, conv_features in
                          zip(many_dense_units, many_conv_features)]

        [model.compile(**compile_params) for model in self.submodels]

        for i, model in enumerate(self.submodels):
            if logging:
                update_log_callback(logging + '/{}/'.format(i + 1), fit_params)
            print_structure(i)
            model.fit(*fit_params[0], **fit_params[1])
        for model in self.submodels:
            model.trainable = False if not trainable else True

        self.Dense1 = layers.Dense(
            2 * self.num_classes * len(self.many_dense_units),
            activation='relu')
        self.dropout_main = layers.Dropout(0.5)
        self.out_layer = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):

        probs = [model(inputs) for model in self.submodels]
        prob_layer = tf.concat(probs, axis=1)
        x = self.Dense1(prob_layer)
        x = self.dropout_main(x)
        return self.out_layer(x)

    def get_model_accuracies(self, x_test, y_test):

        for i, model in enumerate(self.submodels):
            m = keras.metrics.CategoricalAccuracy()
            m.update_state(model(x_test), y_test)
            print(self.many_dense_units[i])
            print(m.result().numpy())
