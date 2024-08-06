#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:08:43 2024

@author: charlene
"""

import import_functions
import helper_functions
import unwrap_functions
import local_SD_analysis_functions
import AI_Methods

from import_functions import get_, pp
from helper_functions import to_dist, plot_dists, helper_cmaps, plot_ims
from unwrap_functions import uw2
from AI_Methods import gen_data_split, normalise, standardise, resize_images, plotHistory

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os

param = 'E2A'
health = 'HCM'
method = 'STEAM'
hs = get_('Healthy', method, 'SYSTOLE', param, 'myo')
hd = get_('Healthy', method, 'DIASTOLE', param, 'myo')
uhs = get_('HCM', method, 'SYSTOLE', param, 'myo')
uhd = get_('HCM', method, 'DIASTOLE', param, 'myo')
hs = pp(hs)
hd = pp(hd)
uhs = pp(uhs)
uhd = pp(uhd)

# Define the generator
def build_generator():
    noise = tf.keras.Input(shape=(100,))
    label = tf.keras.Input(shape=(2,))
    
    model_input = tf.keras.layers.concatenate([noise, label])
    
    x = tf.keras.layers.Dense(128, activation='relu')(model_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64 * 64, activation='sigmoid')(x)
    output = tf.keras.layers.Reshape((64, 64, 1))(x)
    
    model = tf.keras.Model([noise, label], output)
    return model

# Define the discriminator
def build_discriminator():
    image = tf.keras.Input(shape=(64, 64, 1))
    label = tf.keras.Input(shape=(2,))
    
    label_embedding = tf.keras.layers.Dense(64 * 64)(label)
    label_embedding = tf.keras.layers.Reshape((64, 64, 1))(label_embedding)
    
    model_input = tf.keras.layers.concatenate([image, label_embedding])
    
    x = tf.keras.layers.Flatten()(model_input)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model([image, label], output)
    return model

# Compile the cGAN model
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Combine the generator and discriminator into the cGAN model
noise = tf.keras.Input(shape=(100,))
label = tf.keras.Input(shape=(2,))
generated_image = generator([noise, label])
discriminator.trainable = False
validity = discriminator([generated_image, label])
cgan = tf.keras.Model([noise, label], validity)
cgan.compile(optimizer='adam', loss='binary_crossentropy')

# Function to train the cGAN
def train_cgan(generator, discriminator, cgan, data, labels, epochs=2000, batch_size=32):
    half_batch = batch_size // 2
    history = {'d_loss': [], 'g_loss': [], 'd_acc': [], 'val_d_loss': [], 'val_g_loss': [], 'val_d_acc': []}
    
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_images = data[idx]
        real_labels = labels[idx]
        
        noise = np.random.normal(0, 1, (half_batch, 100))
        random_labels = np.random.randint(0, 2, (half_batch, 2))
        fake_images = generator.predict([noise, random_labels])
        
        real_valid = np.ones((half_batch, 1))
        fake_valid = np.zeros((half_batch, 1))
        
        discriminator.trainable = True  # Enable the discriminator
        discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Re-compile the discriminator
        d_loss_real = discriminator.train_on_batch([real_images, real_labels], real_valid)
        d_loss_fake = discriminator.train_on_batch([fake_images, random_labels], fake_valid)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        random_labels = np.random.randint(0, 2, (batch_size, 2))
        valid = np.ones((batch_size, 1))
        
        discriminator.trainable = False  # Freeze the discriminator
        cgan.compile(optimizer='adam', loss='binary_crossentropy')  # Re-compile the cgan
        g_loss = cgan.train_on_batch([noise, random_labels], valid)
        
        # Evaluate the discriminator on real and fake images
        noise = np.random.normal(0, 1, (data.shape[0], 100))
        random_labels = np.random.randint(0, 2, (data.shape[0], 2))
        fake_images = generator.predict([noise, random_labels])
        
        _, acc_real = discriminator.evaluate([data, labels], np.ones((data.shape[0], 1)), verbose=0)
        _, acc_fake = discriminator.evaluate([fake_images, random_labels], np.zeros((data.shape[0], 1)), verbose=0)
        
        d_acc = 0.5 * (acc_real + acc_fake)
        
        val_noise = np.random.normal(0, 1, (data.shape[0], 100))
        val_random_labels = np.random.randint(0, 2, (data.shape[0], 2))
        val_fake_images = generator.predict([val_noise, val_random_labels])
        
        val_d_loss, val_d_acc = discriminator.evaluate([val_fake_images, val_random_labels], np.zeros((data.shape[0], 1)), verbose=0)
        val_g_loss = cgan.evaluate([val_noise, val_random_labels], np.ones((data.shape[0], 1)), verbose=0)
        
        # Store the metrics in the history
        history['d_loss'].append(d_loss[0])
        history['g_loss'].append(g_loss[0] if isinstance(g_loss, list) else g_loss)
        history['d_acc'].append(d_acc)
        history['val_d_loss'].append(val_d_loss)
        history['val_g_loss'].append(val_g_loss[0] if isinstance(val_g_loss, list) else val_g_loss)
        history['val_d_acc'].append(val_d_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, d_loss: {d_loss[0]:.4f}, g_loss: {g_loss[0] if isinstance(g_loss, list) else g_loss:.4f}, d_acc: {d_acc:.4f}, val_d_loss: {val_d_loss:.4f}, val_g_loss: {val_g_loss[0] if isinstance(val_g_loss, list) else val_g_loss:.4f}, val_d_acc: {val_d_acc:.4f}')
   
    return history

h_imgs = hd[0] #retrieve only images not maps
uh_imgs = uhd[0] #retrieve only images not maps
X_train,X_test,y_train,y_test = gen_data_split(h_imgs,uh_imgs,5) #create data split, taking 5 from healthy and 5 from unhealthy
X_train,X_test = normalise(X_train,X_test)
X_train,X_test = resize_images(X_train,X_test)

data = X_train
labels = y_train

# Train the cGAN
history = train_cgan(generator, discriminator, cgan, data, labels, epochs=100, batch_size=32)

# loss = history['d_loss']
# vloss = history['val_d_loss']
# acc = [i*100 for i in history['d_acc']]
# vacc = [i*100 for i in history['val_d_acc']]
# X = [i+1 for i in range(len(vacc))]
   
# fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12,6))
# ax[0].plot(X, acc, '-k',label='Training Accuracy')
# ax[0].plot(X, vacc, '-r', label='Validation Accuracy')
# ax[0].set_title('Accuracy Vs Epoch')
# ax[0].set_xlabel('Epoch',fontsize=18)
# ax[0].set_ylabel('Accuracy (%)',fontsize=18)
# ax[0].tick_params(axis='both', which='major', labelsize=12)
# ax[0].axis('tight')
# ax[0].legend(fontsize=18)
# ax[0].grid('True')
# ax[1].plot(X, loss, '-k', label='Training Loss')
# ax[1].plot(X, vloss, '-r', label='Validation Loss')
# ax[1].set_title('Loss Vs Epoch')
# ax[1].set_xlabel('Epoch',fontsize=18)
# ax[1].set_ylabel('Loss',fontsize=18)
# ax[1].axis('tight')
# ax[1].tick_params(axis='both', which='major', labelsize=12)
# ax[1].legend(fontsize=18)
# ax[1].grid('True')
# plt.show()

def plot_images(real_images, fake_images, num_images=10):
    """
    Plots real and fake images side by side for comparison.

    Parameters:
    - real_images: Numpy array of real images
    - fake_images: Numpy array of fake images generated by cGAN
    - num_images: Number of images to display
    """
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 2))
    for i in range(num_images):
        # Display real images
        axes[i, 0].imshow(real_images[i].reshape(64, 64), cmap='gray')
        axes[i, 0].set_title('Real Image')
        axes[i, 0].axis('off')
        
        # Display fake images
        axes[i, 1].imshow(fake_images[i].reshape(64, 64), cmap='gray')
        axes[i, 1].set_title('Fake Image')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Generate random labels for fake image generation
def generate_random_labels(num_samples):
    """
    Generates random labels for the fake image generation.

    Parameters:
    - num_samples: Number of labels to generate

    Returns:
    - Numpy array of random labels
    """
    return np.random.randint(0, 2, (num_samples, 2))

# Generate fake images
def generate_fake_images(generator, num_samples):
    """
    Generates fake images using the cGAN generator.

    Parameters:
    - generator: The generator model
    - num_samples: Number of fake images to generate

    Returns:
    - Numpy array of generated fake images
    """
    noise = np.random.normal(0, 1, (num_samples, 100))
    labels = generate_random_labels(num_samples)
    fake_images = generator.predict([noise, labels])
    return fake_images

# Choose the number of images to display
num_images_to_display = 10

# Get a subset of real images from the dataset
real_images_subset = X_test[:num_images_to_display]

# Generate fake images using the generator
fake_images_subset = generate_fake_images(generator, num_images_to_display)

# Plot real and fake images
plot_images(real_images_subset, fake_images_subset)

# # Generate images with specific conditions
# def generate_conditioned_images(generator, num_samples, label):
#     """
#     Generates images with a specific condition using the cGAN generator.

#     Parameters:
#     - generator: The generator model
#     - num_samples: Number of images to generate
#     - label: Condition label (e.g., [1, 0] for healthy, [0, 1] for unhealthy)

#     Returns:
#     - Numpy array of generated images
#     """
#     noise = np.random.normal(0, 1, (num_samples, 100))
#     labels = np.array([label] * num_samples)
#     return generator.predict([noise, labels])

# # Classify images using the discriminator
# def classify_images(discriminator, images, labels):
#     """
#     Classifies images using the discriminator.

#     Parameters:
#     - discriminator: The discriminator model
#     - images: Numpy array of images to classify
#     - labels: Numpy array of labels

#     Returns:
#     - Numpy array of predictions
#     """
#     predictions = discriminator.predict([images, labels])
#     return predictions

# # Plot images with their classifications
# def plot_classified_images(images, predictions, num_images=10):
#     """
#     Plots classified images with their predicted labels.

#     Parameters:
#     - images: Numpy array of images
#     - predictions: Numpy array of predictions
#     - num_images: Number of images to display
#     """
#     fig, axes = plt.subplots(num_images, 1, figsize=(5, num_images * 5))
#     for i in range(num_images):
#         axes[i].imshow(images[i].reshape(64, 64), cmap='gray')
#         axes[i].set_title(f'Prediction: {"Healthy" if predictions[i] > 0.5 else "Unhealthy"}')
#         axes[i].axis('off')
#     plt.tight_layout()
#     plt.show()

# # Generate images for both conditions
# num_images_to_generate = 10
# healthy_images = generate_conditioned_images(generator, num_images_to_generate, [1, 0])
# unhealthy_images = generate_conditioned_images(generator, num_images_to_generate, [0, 1])

# # Generate labels for classification
# healthy_labels = np.array([[1, 0]] * num_images_to_generate)
# unhealthy_labels = np.array([[0, 1]] * num_images_to_generate)

# # Classify the generated images
# healthy_predictions = classify_images(discriminator, healthy_images, healthy_labels)
# unhealthy_predictions = classify_images(discriminator, unhealthy_images, unhealthy_labels)

# # Plot classified images
# print("Healthy Images:")
# plot_classified_images(healthy_images, healthy_predictions)

# print("Unhealthy Images:")
# plot_classified_images(unhealthy_images, unhealthy_predictions)

