import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

default_conv_features = [16, 32]


class BaseModel(keras.Model):
    def __init__(self, dense_units, num_classes, conv_features=None):
        super(BaseModel, self).__init__()
        if conv_features is None:
            conv_features = default_conv_features
        self.conv_features = conv_features
        # Start of the model layers:
        self.convs = [layers.Conv2D(c, kernel_size=(3, 3),
                                    activation='relu') for c in conv_features]
        self.batch_norms = [layers.BatchNormalization() for _ in conv_features]
        self.max_pools = [layers.MaxPooling2D(pool_size=(2, 2))
                          for _ in conv_features]
        self.dropouts = [layers.Dropout(0.25) for _ in conv_features]
        self.flat = layers.Flatten()
        self.denses = [layers.Dense(d, activation="relu") for d in dense_units]
        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.5)
        self.probs = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = inputs
        for i in range(len(self.conv_features)):
            x = self.convs[i](x)
            x = self.batch_norms[i](x)
            x = self.max_pools[i](x)
            x = self.dropouts[i](x)
        x = self.flat(x)
        for dense in self.denses:
            x = dense(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return self.probs(x)


class StackingModel(keras.Model):

    def __init__(self, num_classes, compile_params,
                 fit_params, many_dense_units, many_conv_features=None,
                 trainable=False,
                 logging=None, disp_summary=False):

        def update_log_callback(logging, fit_params):
            tensorboard_callback_sub = tf.keras.callbacks.TensorBoard(
                log_dir=logging, histogram_freq=1)
            # Copy fit_params to keep original untouched
            res = [fit_params[0][:], dict(fit_params[1])]
            callbacks = res[1]['callbacks'][:]
            callbacks += [tensorboard_callback_sub]
            res[1]['callbacks'] = callbacks
            print(res[1]['callbacks'])
            return res

        def print_structure(i):
            print('Structure of submodel {}:'.format(i + 1))
            print('\tConv shapes: {}'.format(many_conv_features[i])) if not (
                    many_conv_features[i] is None) else print(
                '\tConv shapes: {}'.format(default_conv_features))
            print('\tDense shapes: {}'.format(many_dense_units[i]))
            return None

        super(StackingModel, self).__init__()
        if many_conv_features is None:
            many_conv_features = [None] * len(many_dense_units)
        assert len(many_dense_units) == len(many_conv_features)

        self.many_dense_units = many_dense_units
        self.num_classes = num_classes
        self.submodels = [BaseModel(dense_units, num_classes, conv_features)
                          for dense_units, conv_features in
                          zip(many_dense_units, many_conv_features)]

        print('Model Pretrain Summary:')
        for i in range(len(self.submodels)):
            print("\tDense Shapes: {} Conv Shapes: {}".format(
                many_dense_units[i], many_conv_features[i]))

        # Compile each submodel
        [model.compile(**compile_params) for model in self.submodels]

        # No list comprehension on the fitting,
        # list can't hold all models in memory.
        for i, model in enumerate(self.submodels):
            submodel_params = fit_params[:]  # Make a copy for each loop
            if logging:
                log_path = logging + '/{}/'.format(i + 1)
                submodel_params = update_log_callback(log_path, fit_params)
            print_structure(i)
            model.fit(*submodel_params[0], **submodel_params[1])
            model.summary() if disp_summary else False
        for model in self.submodels:
            model.trainable = False if not trainable else True

        # Meta Learner Layers
        self.Dense_meta = layers.Dense(
            2 * self.num_classes * len(self.many_dense_units),
            activation='relu')
        self.dropout_meta = layers.Dropout(0.5)
        self.out_layer = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):

        # Feed in images to each base model, concatenate probabilities
        # to feed into meta learner.
        probs = [model(inputs) for model in self.submodels]
        prob_layer = tf.concat(probs, axis=1)
        x = self.Dense_meta(prob_layer)
        x = self.dropout_meta(x)
        return self.out_layer(x)

    def get_model_accuracies(self, x_test, y_test):

    # Currently only used for mnist... Needs adjustments
    # to work with keras ImageDataGenerators
        for i, model in enumerate(self.submodels):
            m = keras.metrics.CategoricalAccuracy()
            m.update_state(model(x_test), y_test)
            print(self.many_dense_units[i])
            print(m.result().numpy())
