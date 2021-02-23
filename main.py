#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
from tqdm import tqdm
import os 
import time
import matplotlib.pyplot as plt


# In[2]:


translate_list = [' ',
             '1',
             '2',
             '3',
             '4',
             '5',
             '6',
             '7',
             '8',
             '9',
             '0',
             '+',
             '-',
             '*',
             '/',
             '=',
            ]


# In[3]:


CHAR_LIST_LEN = len(translate_list)
PROBLEM_LENGTH = 5
PROBLEM_DIM = PROBLEM_LENGTH * CHAR_LIST_LEN

SEED_SIZE = 1000

N_EPOCHS = 20
BATCH_SIZE = 50
BUFFER_SIZE = 60000


# In[4]:


def prob_to_array(in_string):
    problem_array = []
    for char in in_string:
        value = translate_list.index(char)
        char_array = [0]*CHAR_LIST_LEN
        char_array[value] = 1
        problem_array.append(char_array)
    if len(in_string) < PROBLEM_LENGTH:
        for _ in range(PROBLEM_LENGTH-len(in_string)):
            char_array = [0]*CHAR_LIST_LEN
            char_array[0] = 1
            problem_array.append(char_array)
    return problem_array

def array_to_prob(problem_array):
    problem_string = ""
    for char_array in problem_array:
        largest = max(char_array)
        value = char_array.index(largest)
        char = translate_list[value]
        problem_string += char
    problem_string.strip()
    return problem_string

#myarr = prob_to_array('90+21')
#myprob = array_to_prob(myarr)
#print(myarr)
#print(myprob)


# In[5]:


import random
def generate_input(count):
    output_list = []
    for _ in range(count):
        left = random.randint(10,99)
        right = random.randint(10,99)
        problem = str(left) + '+' + str(right)
        output_list.append(problem)
    return output_list


# In[6]:


input_problems = generate_input(1000)
# print(input_problems)
training_data = []
for item in input_problems:
    temp_arr = prob_to_array(item)
    training_data.append(temp_arr)
training_data = np.asarray(training_data)
training_data = training_data.astype(np.float32)
print("Shape: " + str(training_data.shape))


# In[7]:


train_dataset = tf.data.Dataset.from_tensor_slices(training_data)     .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# In[8]:


generator = Sequential([
    Dense(64,activation="relu",input_dim=SEED_SIZE),
    Dense(128, activation="selu"),
    #model.add(Dense(28*28, activation="sigmoid")),
    Dense(PROBLEM_LENGTH*CHAR_LIST_LEN, activation="sigmoid"),
    Reshape([PROBLEM_LENGTH, CHAR_LIST_LEN])
])

discriminator = Sequential([
    Flatten(input_shape=[PROBLEM_LENGTH, CHAR_LIST_LEN]),
    Dense(128, activation="selu"),
    Dense(64, activation="selu"),
    Dense(1, activation="sigmoid")
])


# In[9]:


discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
gan = Sequential([generator, discriminator])
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")


# In[10]:


def display_problems(fake_problems):
    numpy_array = fake_problems.numpy()
    # print(numpy_array)
    for prob_array in numpy_array:
        py_array = prob_array.tolist()
        prob_string = array_to_prob(py_array)
        print(prob_string)
    # print(fake_problems.shape)


# In[11]:



"""
Generator not random enough
Sometimes there is one character that persists when it shouldnt
Maybe increase size of random input
"""
for epoch in range(N_EPOCHS):
    for real_problems in train_dataset:
        print("------------------------------")
        # print("Real Problems: " + str(real_problems.shape))
        noise = tf.random.normal(shape=[BATCH_SIZE, SEED_SIZE])
        # print("Noise: " + str(noise.shape))
        fake_problems = generator(noise)
        #print("Real: ")
        #display_problems(real_problems)
        print("Fakes: ")
        display_problems(fake_problems)
        mixed_problems = tf.concat([fake_problems, real_problems], axis=0)
        # print("Mixed problems: " + str(mixed_problems.shape))
        discriminator_labels = tf.constant([[0.]] * BATCH_SIZE + [[1.]] * BATCH_SIZE)
        # print("Discrim labels: " + str(discriminator_labels.shape))
        discriminator.trainable = True
        print("Discriminator loss: " + str(discriminator.train_on_batch(mixed_problems, discriminator_labels)))
        #discriminator.train_on_batch(mixed_problems, discriminator_labels)
        #print("Discriminator summary: " + str(discriminator.summary()))
        noise = tf.random.normal(shape=[BATCH_SIZE, SEED_SIZE])
        generator_labels = tf.constant([[1.]] * BATCH_SIZE)
        discriminator.trainable = False
        print("Generator loss: " + str(gan.train_on_batch(noise, generator_labels)))
        #gan.train_on_batch(noise, generator_labels)
        # print("GAN summary: " + str(gan.summary()))


# In[ ]:




