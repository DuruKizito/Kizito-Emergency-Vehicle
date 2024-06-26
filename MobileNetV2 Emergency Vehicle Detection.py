#!/usr/bin/env python
# coding: utf-8

# In[22]:


# Import the necessary libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# In[23]:


# Load the images and labels into a dataframe
data = []
labels = []
image_size = 224
for folder in ['emergency', 'non-emergency']:
    label = 1 if folder == 'emergency' else 0 
    for file in glob.glob(f'{folder}/*.jpg'): 
        image = cv2.imread(file) 
        image = cv2.resize(image, (image_size, image_size)) 
        data.append(image) 
        labels.append(label) 
data = np.array(data) / 255.0 # scale the pixel values to [0, 1]
labels = np.array(labels) 
df = pd.DataFrame({'data': list(data), 'labels': labels}) 
df.head()


# In[24]:


# Split the dataframe into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['data'].tolist(), df['labels'].tolist(), test_size=0.2, shuffle=True, random_state=42) 
X_train = np.array(X_train) 
X_test = np.array(X_test) 
y_train = np.array(y_train) 
y_test = np.array(y_test) 
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[25]:


# Define the CNN architecture
base_model = tf.keras.applications.MobileNetV2(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet') 
base_model.trainable = True 
model = tf.keras.models.Sequential() 
model.add(base_model) 
model.add(tf.keras.layers.GlobalAveragePooling2D()) 
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # use sigmoid for binary output
model.add(tf.keras.layers.Dropout(0.2)) 
model.summary()


# In[26]:


# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()]) # use binary crossentropy and binary accuracy for binary classification


# In[27]:


# Train the model
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_binary_accuracy', save_best_only=True, mode='max') # use val_binary_accuracy as the monitor
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) 
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint, earlystop])


# In[28]:


# Evaluate the model
y_pred = model.predict(X_test) # get the probabilities
y_pred = np.where(y_pred > 0.5, 1, 0) # apply a threshold
acc = accuracy_score(y_test, y_pred) # calculate accuracy
prec = precision_score(y_test, y_pred) # calculate precision
rec = recall_score(y_test, y_pred) # calculate recall
f1 = f1_score(y_test, y_pred) # calculate F1-score
cm = confusion_matrix(y_test, y_pred) # calculate confusion matrix
print(f'Accuracy: {acc:.2f}')
print(f'Precision: {prec:.2f}')
print(f'Recall: {rec:.2f}')
print(f'F1-score: {f1:.2f}')
print(f'Confusion matrix:\n{cm}')


# In[32]:


# Label an image from the code
img = cv2.imread('79.jpg') # read the image
img = cv2.resize(img, (224, 224)) # resize the image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert the color
plt.imshow(img) # show the image
pred = model.predict(np.expand_dims(img, axis=0)) # get the probability
label = 1 if pred > 0.5 else 0 # apply a threshold
bbox = (0, 0, 224, 224) # use the whole image as the bounding box
classLabels = ['emergency', 'non-emergency'] # define the class names
cv2.rectangle(img, bbox, (255, 0, 0), 2) # draw a rectangle
cv2.putText(img, classLabels[label], (bbox[0]+10, bbox[1]+40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3) # draw a text
plt.imshow(img) # show the labeled image


# In[34]:


# Video Demo
cap = cv2.VideoCapture('Emergency.mp4') # Use 0 for webcam
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
classLabels = ['emergency', 'non-emergency'] # define the class names

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (image_size, image_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0 # scale the pixel values to [0, 1]
    pred = model.predict(np.expand_dims(frame, axis=0)) # get the probability
    label = 1 if pred > 0.5 else 0 # apply a threshold
    bbox = (0, 0, image_size, image_size) # use the whole image as the bounding box
    
    cv2.rectangle(frame, bbox, (255, 0, 0), 2) # draw a rectangle
    cv2.putText(frame, classLabels[label], (bbox[0]+10, bbox[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3) # draw a text
                
    cv2.imshow('Emergency Vehicle Detection', frame)
                
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




