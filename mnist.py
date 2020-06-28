from ensemblenet import EnsembleModel
import tensorflow.keras as keras
import numpy as np

np.random.seed(0)

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
shapes = list(map(tuple, np.random.randint(200, 2000, (3, 3))))
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
     'epochs': 10,
     'validation_split': 0.1}
)

model = EnsembleModel(shapes, num_classes, compile_params, fit_params)
model.compile(**compile_params)
model.fit(*fit_params2[0], **fit_params2[1])

# Model Assessment
model.get_model_accuracies(x_test, y_test)
m = keras.metrics.CategoricalAccuracy()
m.update_state(model(x_test), y_test)
print("Full model.")
print(m.result().numpy())
