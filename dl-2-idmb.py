#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[70]:


from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000) # you may take top 10,000
#word frequently used review of movies other are discarded
#consolidating data for EDA Exploratory data analysis (EDA) is used by data scientists to analyze and
#investigate data sets and summarize their main characteristics
data = np.concatenate((X_train, X_test), axis=0) # axis 0 is first running vertically downwards across
#rows (axis 0), axis 1 is second running horizontally across columns (axis 1),
label = np.concatenate((y_train, y_test), axis=0)


# In[71]:


X_train.shape


# In[72]:


X_test.shape


# In[73]:


y_train.shape


# In[74]:


y_test.shape


# In[75]:


print("Review is ",X_train[0]) # series of no converted word to vocabulory associated with index
print("Review is ",y_train[0])


# In[76]:


vocab=imdb.get_word_index() # Retrieve the word index file mapping words to indices
print(vocab)


# In[77]:


y_train


# In[78]:


y_test


# In[79]:


def vectorize(sequences, dimension = 10000): # We will vectorize every review and fill it with zeros
#so that it contains exactly 10,000 numbers.
# Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


# In[80]:


test_x = data[:10000]
test_y = label[:10000]
train_x = data[10000:]
train_y = label[10000:]
test_x.shape


# In[81]:


test_y.shape


# In[82]:


train_x.shape


# In[83]:


train_y.shape


# In[84]:


print("Categories:", np.unique(label))
print("Number of unique words:", len(np.unique(np.hstack(data))))


# In[85]:


length = [len(i) for i in data]
print("Average Review length:", np.mean(length))
print("Standard Deviation:", round(np.std(length)))


# In[86]:


print("Label:", label[0])


# In[87]:


print("Label:", label[1])


# In[88]:


print(data[0])


# In[89]:


index = imdb.get_word_index() 


# In[90]:


reverse_index = dict([(value, key) for (key, value) in index.items()]) 


# In[91]:


decoded = " ".join( [reverse_index.get(i - 3, "#") for i in data[0]] )


# In[92]:


print(decoded)


# In[93]:


import seaborn as sns


# In[94]:


data = vectorize(data)
label = np.array(label).astype("float32")
labelDF=pd.DataFrame({'label':label})
sns.countplot(x='label', data=labelDF)


# In[95]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,label, test_size=0.20, random_state=1)
X_train.shape
(40000, 10000)
X_test.shape
(10000, 10000)
# Let's create sequential model
from keras.utils import to_categorical
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(50, activation = "relu", input_shape=(10000, )))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
# Output- Layer
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()


# In[96]:


import tensorflow as tf
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model.compile(
optimizer = "adam",
loss = "binary_crossentropy",
metrics = ["accuracy"]
)
from sklearn.model_selection import train_test_split
results = model.fit(
X_train, y_train,
epochs= 2,
batch_size = 500,
validation_data = (X_test, y_test),
callbacks=[callback]
)
# Let's check mean accuracy of our model
print(np.mean(results.history["val_accuracy"]))
# Evaluate the model
score = model.evaluate(X_test, y_test, batch_size=500)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[97]:


print(results.history.keys())


# In[98]:


import matplotlib. pyplot as plt


# In[99]:


plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[100]:


# summarize history for loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:




