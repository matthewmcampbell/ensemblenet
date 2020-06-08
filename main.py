import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np

num_classes = 10
# Load data
((x_train, y_train),
	(x_test, y_test)) = keras.datasets.mnist.load_data()

(x_train, x_test) = map(
	lambda data: data.astype("float32")/255, 
	(x_train, x_test))

(x_train, x_test) = map(
	lambda data: np.expand_dims(data, -1),
	(x_train, x_test))

(y_train, y_test) = map(
	lambda data: keras.utils.to_categorical(data, num_classes),
	(y_train, y_test))
test = np.expand_dims(x_train[0,:,:,:], 0)
print("x_train shape:", x_train.shape)
print("Train Samples: ", x_train.shape[0])
print("Test Samples: ", x_test.shape[0])

input_shape = (28, 28, 1)
model = keras.Sequential(
	[
		keras.Input(shape=input_shape),
		layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
		layers.MaxPooling2D(pool_size=(2, 2)),
		layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
		layers.MaxPooling2D(pool_size=(2, 2)),
		layers.Flatten(),
		layers.Dense(30, activation="sigmoid"),
		layers.Dense(50, activation="relu"),
		layers.Dropout(0.5),
		layers.Dense(num_classes, activation="softmax")
	]
)

model.summary()

batch_size = 1000
epochs = 1
compile_params = {
	'loss': "categorical_crossentropy",
	'optimizer': 'adam',
	'metrics': ['accuracy']
}
model.compile(**compile_params)

fit_params = (
	[x_train, y_train],
	{'batch_size': batch_size,
	'epochs': epochs,
	'validation_split': 0.1})
# model.fit(*fit_params[0], **fit_params[1])

# print([l.weights for l in model.layers])

class EnsembleLayer(layers.Layer):
	def __init__(self, dense_units):
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

	# def build(self, input_shape):
	# 	self.w = self.add_weight(
	# 		shape=(input_shape[-1], 32),
	# 		initializer = "random_normal",
	# 		trainable = True)
	# 	self.b = self.add_weight(
	# 		shape = (32, ), initializer="random_normal", trainable = True)

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

# model = keras.Sequential([
# 		keras.Input(shape = input_shape),
# 		EnsembleLayer([30, 50])
# 		])
# model.summary()
# model.compile(**compile_params)
# model.fit(*fit_params[0], **fit_params[1])
class EnsembleUnit(keras.Model):
	def __init__(self, dense_units):
		super(EnsembleUnit, self).__init__()
		self.block = EnsembleLayer(dense_units)

	def call(self, inputs):
		return self.block(inputs)

class EnsembleModel(keras.Model):

	def __init__(self, many_dense_units, inputs, outputs, epochs):
		super(EnsembleModel, self).__init__()
		self.many_dense_units = many_dense_units
		compile_params = {
			'loss': "categorical_crossentropy",
			'optimizer': 'adam',
			'metrics': ['accuracy']
		}
		# model.compile(**compile_params)

		fit_params = (
			[inputs, outputs],
			{'batch_size': batch_size,
			'epochs': epochs,
			'validation_split': 0.1})

		self.submodels = [EnsembleUnit(dense_units) for dense_units in many_dense_units]
		[model.compile(**compile_params) for model in self.submodels]
		[model.fit(*fit_params[0], **fit_params[1]) for model in self.submodels]
		for model in self.submodels:
			model.trainable = False
		self.out_layer = layers.Dense(10, activation = "softmax")

	def call(self, inputs):
		probs = [model(inputs) for model in self.submodels]
		prob = tf.concat(probs, axis=1)
		return self.out_layer(prob)

	def get_model_accuracies(self, x_test, y_test):
		for i, model in enumerate(self.submodels):
			m = keras.metrics.Accuracy()
			m.update_state(model(x_test), y_test)
			print(self.many_dense_units[i])
			print(m.result().numpy())

# model = EnsembleUnit([30, 60])
# model.build(input_shape = input_shape)
# model.compile(**compile_params)
# model.fit(*fit_params[0], **fit_params[1])
fit_params2 = (
			[x_train, y_train],
			{'batch_size': batch_size,
			'epochs': 1,
			'validation_split': 0.1})

shapes = list(map(tuple, np.random.randint(1600, 2000, (3,2))))
shapes = list(set(shapes)) #use unique vals only
model = EnsembleModel(shapes, x_train, y_train, 1)
model.compile(**compile_params)
model.fit(*fit_params[0], **fit_params2[1])

model.get_model_accuracies(x_test, y_test)
m = keras.metrics.Accuracy()
m.update_state(model(x_test), y_test)
print("Full model.")
print(m.result().numpy)
# print(model.predict(test).shape)
# print(model(test).shape)
model.summary()