import tensorflow.keras as keras
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os

from ensemblenet import EnsembleModel

np.random.seed(0)
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
path = 'c:/users/matthew/projects/data/dogs-vs-cats/train/train/'
filenames = os.listdir(path)

categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)
file_df = pd.DataFrame({'filename': filenames, 'category': categories})
file_df['category'].replace({0: 'cat', 1: 'dog'}, inplace=True)
file_df = file_df.sample(frac = 0.05)
train_df, val_df = train_test_split(file_df, test_size=0.2, random_state=0)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

total_train = train_df.shape[0]
total_validate = val_df.shape[0]
batch_size=32

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    path, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=32
)
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_dataframe(
    val_df, 
    path, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=32
)

num_classes = 2
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
    [train_generator], 
    {'epochs': 10,
    'validation_data': val_generator,
    'validation_steps': total_validate//batch_size,
    'steps_per_epoch': total_train//batch_size,
    # 'callbacks': callbacks,
    }
    )
fit_params2 = (
    [train_generator],
    {'epochs': 10,
    'validation_data': val_generator,
    'validation_steps': total_validate//batch_size,
    'steps_per_epoch': total_train//batch_size,
    # 'callbacks': callbacks,
    }
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