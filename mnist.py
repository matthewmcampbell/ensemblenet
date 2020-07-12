from ensemblenet import StackingModel
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, Session, RunOptions
from tensorflow.compat.v1.keras.backend import set_session

# Run Options
np.random.seed(0)  # Seed for reproducing
tf.keras.fit_verbose = 2  # Print status after every epoch
RunOptions.report_tensor_allocations_upon_oom = True  # Out-of-memory display
config = ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow GPU memory
config.log_device_placement = True
sess = Session(config=config)
set_session(sess)

num_classes = 10
# Load data and clean
((x_train, y_train),
 (x_test, y_test)) = keras.datasets.mnist.load_data()

(x_train, x_test) = map(
    lambda data: data.astype("float32") / 255,
    (x_train, x_test))

(x_train, x_test) = map(
    lambda data: np.expand_dims(data, -1),
    (x_train, x_test))

(y_train, y_test) = map(
    lambda data: keras.utils.to_categorical(data, num_classes),
    (y_train, y_test))

test = np.expand_dims(x_train[0, :, :, :], 0)
print("x_train shape:", x_train.shape)
print("Train Samples: ", x_train.shape[0])
print("Test Samples: ", x_test.shape[0])

# input_shape = (28, 28, 1)

compile_params = {
    'loss': "categorical_crossentropy",
    'optimizer': 'adam',
    'metrics': ['accuracy']
}
shapes = tuple(map(lambda x: sorted(tuple(x)), np.random.randint(100, 500,
                                                                (3, 3))))
conv_features = (
    (16, 32),
    (8, 16),
    (8, 32)
)
compile_params = {
    'loss': "categorical_crossentropy",
    'optimizer': 'RMSprop',
    'metrics': ['accuracy']
}
fit_params = (
    [x_train, y_train],
    {'batch_size': 100,
     'epochs': 3,
     'validation_split': 0.1}
)
fit_params2 = (
    [x_train, y_train],
    {'batch_size': 100,
     'epochs': 2,
     'validation_split': 0.1}
)

model = StackingModel(num_classes, compile_params, fit_params, shapes,
                      conv_features)
model.compile(**compile_params)
model.fit(*fit_params2[0], **fit_params2[1])

# Model Assessment
model.get_model_accuracies(x_test, y_test)
m = keras.metrics.CategoricalAccuracy()
m.update_state(model(x_test), y_test)
print("Full model.")
print(m.result().numpy())
