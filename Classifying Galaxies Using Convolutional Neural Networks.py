import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from utils import load_galaxy_data


input_data, labels = load_galaxy_data()

# Dimensions of out Input Data and Labels
print('Dimensions of Input Data: ', input_data.shape)
print('Dimensions of Labels: ', labels.shape)


# Splitting the data into x/y test/train
x_train, x_test, y_train, y_test = train_test_split(input_data, labels, test_size = 0.20, shuffle = True, random_state = 222, stratify = labels)


# Preprocess the input data using ImageDataGenerator (normalise pixels and rescale parameters)
data_generator = ImageDataGenerator(rescale = 1.0/255)

# Creating 2 numpy array iterators
training_iterator = data_generator.flow(x_train, y_train, batch_size = 5)
testing_iterator = data_generator.flow(x_test, y_test, batch_size = 5)

# Building the Model
model = tf.keras.Sequential()
# Adding the input layer with the shape of 128, 128, 3
model.add(tf.keras.Input(shape = input_data.shape[1:]))
# Adding first Convolutional layer, 8 filters of 3x3 with strides of 2
model.add(tf.keras.layers.Conv2D(8, 3, strides = 2, activation = 'relu'))
# Adding first pooling layer of size 2 and strides 2
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))
# Adding second Convolutional layer, 8 filters of 3x3 with strides of 2
model.add(tf.keras.layers.Conv2D(8, 3, strides = 2, activation = 'relu'))
# Adding the second pooling layer of size 2 and strides 2
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))
# Adding the flatten layer
model.add(tf.keras.layers.Flatten())
# Adding our hidden dense layer of 16 hidden units
model.add(tf.keras.layers.Dense(16, activation = 'relu'))
# Adding the output layer with 4 outputs and a softmax activation function
model.add(tf.keras.layers.Dense(4, activation = 'softmax'))

# Compiling the model
model.compile(
  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
  loss = tf.keras.losses.CategoricalCrossentropy(),
  metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]
)

# Printing model summary
print(model.summary())

# Training the model
model.fit(
  training_iterator,
  steps_per_epoch = len(x_train) / 5, # length of training data divided by batch size
  epochs = 12,
  validation_data = testing_iterator, # length of testing data divided by batch size
  validation_steps = len(x_test) / 5 
)

""" 
A random baseline model would achieve only ~25% accuracy on the dataset.
This model achieved a ~70%
The AUC tells you that for a random galaxy, there is more than an 80% chance the model 
would assign a higher probability to a true class than to a false one.
"""