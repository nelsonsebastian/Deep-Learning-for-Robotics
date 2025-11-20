from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Display an image
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),#MNIST images are 28×28, grayscale with 1 channel,32 filters(32 differnt 3*3)
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])#
model.summary()

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test)) #after each epoch, the model checks its performance on unseen test data(test data)
model.evaluate(x_test, y_test)
# Save full model (architecture + weights)
model.save('digit_classifier.keras')

# Save only weights
model.save_weights('digit_classifier.weights.h5')

import numpy as np
from PIL import Image
from tensorflow import keras

model = keras.models.load_model('digit_classifier.keras')

model.load_weights('digit_classifier.weights.h5')

prediction_image = Image.open('/content/seven.png').convert('L')  # Grayscale
prediction_image = prediction_image.resize((28, 28))  # standard input size for the MNIST dataset that this type of digit classifier model is trained on.

prediction_image = np.array(prediction_image)
prediction_image = prediction_image.reshape(1, 28, 28, 1).astype('float32') / 255.0 #predicting one image at a time,width,height,channelone gray

predictions = model.predict(prediction_image)
print(predictions)
predicted_class = np.argmax(predictions)

print(f"The predicted digit is: {predicted_class}")
