import numpy as np
import mnist

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

print(train_images.shape) # (60000, 784)
print(test_images.shape)  # (10000, 784)

# first is input, last is output
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'), 
])

train = True
test = True

if train:
	model.compile(
	  optimizer=Adam(lr=0.005),
	  loss='categorical_crossentropy',
	  metrics=['accuracy'],
	)

	model.fit(
	  train_images, # training data
	  to_categorical(train_labels), # training targets
	  epochs=5,
	  batch_size=32,
	)

	model.evaluate(
	  test_images,
	  to_categorical(test_labels)
	)

	model.save_weights('model.h5')

if test:
	# Load the model's saved weights.
	model.load_weights('model.h5')

	# Predict on the first 5 test images.
	predictions = model.predict(test_images[:5])

	# Print our model's predictions.
	print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

	# Check our predictions against the ground truths.
	print(test_labels[:5]) # [7, 2, 1, 0, 4]
