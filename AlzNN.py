import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf

import cv2
import os

#make a dataframe from a folder of images
import os
import cv2
import numpy as np
from IPython.display import display, clear_output
labels = ['MildDemented', 'ModerateDemented','NonDemented','VeryMildDemented']
img_size = 112

# This function takes folders of images and turns it into an array of images that the model can iterate over
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for x,img in enumerate(os.listdir(path)):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
            clear_output(wait=True)
    return np.array(data)

# Uncomment the below section if it is your first time running -- Line 38 & 39 execute the function from line 21
# Line 42-45: saving the numpy array 
"""train = get_data("/Users/shivani/Documents/ML_Project/Alzheimers_Dataset/train")
test = get_data("/Users/shivani/Documents/ML_Project/Alzheimers_Dataset/test")

with open('/Users/shivani/Documents/ML_Project/Alzheimers_Dataset/train.npy', 'wb') as f:
    np.save(f,train)
with open('/Users/shivani/Documents/ML_Project/Alzheimers_Dataset/test.npy', 'wb') as f1:
    np.save(f1,test)
    """

# This can load a numpy array directly so you don't have to read every image from storage to memory everytime
with open('/Users/shivani/Documents/ML_Project/Alzheimers_Dataset/train.npy', 'rb') as f2:
    train = np.load(f2,allow_pickle=True)
with open('/Users/shivani/Documents/ML_Project/Alzheimers_Dataset/test.npy', 'rb') as f3:
    test = np.load(f3,allow_pickle=True)
    
# print(train.shape, test.shape)

# Imports
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Dropout
from tensorflow.keras.utils import to_categorical

#1st layer
model = Sequential()
model.add(Conv2D(16,3,padding="same", activation="relu", input_shape=(img_size,img_size,3)))
model.add(MaxPool2D())

#2nd layer
model.add(Conv2D(16, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

#3rd layer
model.add(Conv2D(16, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

#4th layer
model.add(Conv2D(16, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

#5th layer
model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())

#6th layer MLP
model.add(Dense(32,activation="relu"))

#output and 7th layer MLP
model.add(Dense(4, activation="softmax"))

model.summary()

opt = Adam(learning_rate=0.0001)
model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in test:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', save_best_only=True, verbose = 1)

epochs = 75
history = model.fit(x_train, y_train, epochs = epochs , validation_data = (x_val, y_val),callbacks=[mc])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

#Load best model
#model = load_model('best_model.h5')

predictions = model.predict(x_val)
np.save("pred.npy",predictions)
np.save("y_val.npy",y_val)
predictions = np.argmax(predictions, axis=1)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['MildDemented', 'ModerateDemented','NonDemented','VeryMildDemented']))

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
