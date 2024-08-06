#AI Functions
import import_functions
import helper_functions
import unwrap_functions
import local_SD_analysis_functions

from import_functions import get_, pp
from helper_functions import to_dist, plot_dists, helper_cmaps, plot_ims
from unwrap_functions import uw2,pick_centre_middle_
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import random
import pandas as pd
# from local_SD_analysis_functions import calc_var
from sklearn.model_selection import train_test_split
import pickle
import os
import cv2

def gen_data_split(healthy_images, unhealthy_images, test_size, repeats=False):
    if test_size == 0:
        img_h_train = np.array(healthy_images, dtype=object)
        img_uh_train = np.array(unhealthy_images, dtype=object)
        X_train = np.concatenate((img_h_train, img_uh_train))
        y_train = np.concatenate((np.ones((len(img_h_train), 1)), np.zeros((len(img_uh_train), 1))))
        X_train = np.array([img[:, :, None] for img in X_train], dtype=object)
        y_train = tf.keras.utils.to_categorical(y_train)
        return X_train, [], y_train, []
    
    img_h_train, img_h_test = train_test_split(np.array(healthy_images, dtype=object), test_size=test_size, random_state=random.randint(1, 1000))
    img_uh_train, img_uh_test = train_test_split(np.array(unhealthy_images, dtype=object), test_size=test_size, random_state=random.randint(1, 1000))
    
    if repeats:
        img_uh_train = np.repeat(img_uh_train, repeats, axis=0)
    
    X_train = np.concatenate((img_h_train, img_uh_train))
    X_test = np.concatenate((img_h_test, img_uh_test))
    
    y_train = np.concatenate((np.ones((len(img_h_train), 1)), np.zeros((len(img_uh_train), 1))))
    y_test = np.concatenate((np.ones((len(img_h_test), 1)), np.zeros((len(img_uh_test), 1))))
    
    X_train = np.array([img[:, :, None] for img in X_train], dtype=object)
    X_test = np.array([img[:, :, None] for img in X_test], dtype=object)
    
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    
    return X_train, X_test, y_train, y_test



def normalise(healthy_images,unhealthy_images,minn=-90,maxx=90):
    '''
    normalises values to range of -1 to 1 for each image. If angle parameter, dont pass values into minn or maxx.
    If you want it to be automtically selected, set minn,maxx == None

    Inputs:
    healthy_images,unhealthy_images: list of healthy and HCM images
    minn,maxx: values in the original images mapped to -1 and 1

    '''
    imgs_h = []
    imgs_uh = []
    for img in healthy_images:
        img = img.copy()
        if minn is None:
            # img[np.where(img>=4)] = 0 #can be used to remove any anomalous values, MD > 4 is a bit unrealistic and is probably noise.
            minn1 = np.nanmin(img)
            maxx1 = np.nanmax(img)
            mask = np.where(img==0)
            img[np.where(img>=2.5)] = 0
            new_img = 2*(img-minn1)/(maxx1-minn1)-1
            new_img = np.asarray(new_img).astype('float32')
            new_img[np.where(new_img==new_img[0][0])] = 0
            new_img[mask] = 0
            imgs_h.append(new_img)
        else:
            new_img = 2*(img-minn)/(maxx-minn)-1
            new_img = np.asarray(new_img).astype('float32')
            imgs_h.append(new_img)
    for img in unhealthy_images:
        img = img.copy()
        if minn is None:
            # img[np.where(img>=4)] = 0
            minn2 = np.nanmin(img)
            maxx2 = np.nanmax(img)
            mask = np.where(img==0)
            new_img = 2*(img-minn2)/(maxx2-minn2)-1
            new_img = np.asarray(new_img).astype('float32')
            new_img[np.where(new_img==new_img[0][0])] = 0
            new_img[mask] = 0
            imgs_uh.append(new_img)
        else:
            new_img = 2*(img-minn)/(maxx-minn)-1
            new_img = np.asarray(new_img).astype('float32')
            imgs_uh.append(new_img)
    return imgs_h,imgs_uh

def standardise(healthy_images,unhealthy_images):
    ''''
    standardises images using zscore formula
    z = (x-mean)/sd

    '''
    imgs_h = []
    imgs_uh = []

    for img in healthy_images:
        img = img.copy()
        img[np.where(img>=4)] = 0
        mask = np.where(img==0)
        img[mask] = np.nan #set to nan so that its not counted in the mean or var calculation
        new_img = (img-np.nanmean(img))/np.sqrt(np.nanvar(img))
        new_img[np.where(new_img==new_img[0][0])] = np.nan
        new_img[np.isnan(new_img)] = 0
        imgs_h.append(new_img)

    for img in unhealthy_images:
        img = img.copy()
        img[np.where(img>=4)] = 0
        mask = np.where(img==0)
        img[mask] = np.nan
        new_img = (img-np.nanmean(img))/np.sqrt(np.nanvar(img))
        new_img[np.where(new_img==new_img[0][0])] = np.nan
        new_img[np.isnan(new_img)] = 0
        imgs_uh.append(new_img)

    return imgs_h,imgs_uh
def resize_images(X_train,X_test):
    '''
    Resizes the images to a constant size. The largest image had a dimension of 63, so images were chosen
    to be resized to 64 by 64.
    
    Inputs:
    X_train,X_test: list of input data.
    Ouputs:
    new_X_train,new_X_test: numpy arrays of size Mx64x64x1, where M is the number of individuals in
                            training and test set
    '''
    max_height = 64
    max_width = 64
    new_X_train = []
    for image in X_train:
        pad_height = max_height - image.shape[0]
        pad_width = max_width - image.shape[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_image = tf.image.pad_to_bounding_box(image, pad_top, pad_left, max_height, max_width)
        padded_image = padded_image.numpy().copy()
        # padded_image[np.where(padded_image==0)] = -1 #this is where you set what you want the region outside of the LV to be
        new_X_train.append(padded_image)

    new_X_test = []
    for image in X_test:
        pad_height = max_height - image.shape[0]
        pad_width = max_width - image.shape[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_image = tf.image.pad_to_bounding_box(image, pad_top, pad_left, max_height, max_width)
        padded_image = padded_image.numpy().copy()
        # padded_image[np.where(padded_image==0)] = -1
        new_X_test.append(padded_image)
    return np.array(new_X_train),np.array(new_X_test)


def build_cnn(patience,start_from):
    '''
    choose your model architecture here

    Inputs:
    patience: how many epochs the model waits before stopping if it hasnt reached a new minimum of validation loss
    start_from: the number of epochs to wait before starting early stopping criterion

    Outputs:
    model: Compiled TensorFlow Model
    callback: Callback to use during training
    '''
    # tf.keras.utils.disable_interactive_logging()
    input_shape = (64,64,1)
    #callback. Stop if the validation loss doesnt reach a new minimum in patience epochs, starting from start_from
    callback =tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min',start_from_epoch=start_from)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(1, 5, activation='sigmoid', kernel_initializer='he_uniform', input_shape=input_shape,name='Conv2D'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2),name='MaxPooling2D'))
    # model.add(tf.keras.layers.Conv2D(1, 3, activation='sigmoid', kernel_initializer='he_uniform', input_shape=input_shape,name='Conv2D_2'))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2),name='MaxPooling2D_2'))
    model.add(tf.keras.layers.Flatten(name='Flatten'))
    # model.add(tf.keras.layers.Dense(2, activation='sigmoid', kernel_initializer='he_uniform',name='Dense'))
    model.add(tf.keras.layers.Dense(2, activation='softmax',name='Output'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model, callback

    
def train_cnn(model, callback, data, epochs):
    '''trains model and evalutes
    
    Inputs:
    model: Compiled TensorFlow Model
    callback: Callback to use during training
    data: [X_train,y_train,X_test,y_test]
    epochs: number of cycles to train the neural network. Effectively how many times it sees the data. 

    Outputs:
    history: Training and Validation Loss and Accuracy
    info_df: Information dataframe containing final test and train losses and accuracies
    '''
    
    X_train, y_train, X_test, y_test = data

    history = model.fit(X_train,y_train, epochs=epochs, callbacks=[callback],validation_data=(X_test,y_test))
    test_loss, test_acc = model.evaluate(X_test,y_test)
    train_loss, train_acc = model.evaluate(X_train,y_train)
    
    test_loss2 = np.round(test_loss,5)
    train_loss2 = np.round(train_loss,5)
    test_acc2 = 100*test_acc
    train_acc2 = 100*train_acc
    info_df = pd.DataFrame([[test_loss2,test_acc2],[train_loss2,train_acc2]],index=['test','train'], columns=['loss','acc'])
    return history,info_df

    
def plot_history(history):
    '''
    pass in history from run_model to plot accuracy and loss vs epoch for training and validation
    '''
    history = history.history
    loss = history['loss']
    vloss = history['val_loss']
    acc = [i*100 for i in history['accuracy']]
    vacc = [i*100 for i in history['val_accuracy']]
    X = [i+1 for i in range(len(vacc))]
    
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12,6))
    ax[0].plot(X, acc, '-k',label='Training Accuracy')
    ax[0].plot(X, vacc, '-r', label='Validation Accuracy')
    ax[0].set_title('Accuracy Vs Epoch')
    ax[0].set_xlabel('Epoch',fontsize=18)
    ax[0].set_ylabel('Accuracy (%)',fontsize=18)
    ax[0].tick_params(axis='both', which='major', labelsize=12)
    ax[0].axis('tight')
    ax[0].legend(fontsize=18)
    ax[0].grid('True')
    ax[1].plot(X, loss, '-k', label='Training Loss')
    ax[1].plot(X, vloss, '-r', label='Validation Loss')
    ax[1].set_title('Loss Vs Epoch')
    ax[1].set_xlabel('Epoch',fontsize=18)
    ax[1].set_ylabel('Loss',fontsize=18)
    ax[1].axis('tight')
    ax[1].tick_params(axis='both', which='major', labelsize=12)
    ax[1].legend(fontsize=18)
    ax[1].grid('True')
    plt.show()

def save_model(name,model,data,history):
    '''
    saves the model, training and validation data, and training history
    Inputs:
    name: name of model to save. Will be save in ./Models/name/
    model: TensorFlow model to save
    data: training and test data used
    history: training and validation accuracy and loss during training

    '''
    if not os.path.exists(f'./Models/{name}'):
        os.makedirs(f'./Models/{name}')
    with open(f'./Models/{name}/{name}_data.pkl', 'wb') as file: 

        pickle.dump(data, file)
    with open(f'./Models/{name}/{name}_history.pkl', 'wb') as file: 
      
        pickle.dump(history, file)
    model.save(f'./Models/{name}/{name}_model.keras')

def load_model(name):
    '''
    loads the model, training and validation data, and training history
    Inputs:
    name: name of model to load. Will be loaded from ./Models/name/

    Outputs:
    model: TensorFlow model to load
    data: training and test data used to train model
    history: training and validation accuracy and loss during training
    '''
    #new_model = tf.keras.models.load_model(f'./Models/{name}/{name}_model.keras')
    dir = "/Users/charlene/Desktop/UROP/Code"
    new_model = tf.keras.models.load_model(f'{dir}/Models/{name}/{name}_model.keras')
    with open(f'{dir}/Models/{name}/{name}_history.pkl', 'rb') as file: 
      
    # Call load method to deserialze 
        history = pickle.load(file) 

    plot_history(history)
    with open(f'{dir}/Models/{name}/{name}_data.pkl', 'rb') as file: 

        data = pickle.load(file)
    return new_model,data,history

def predict_and_visualise(model, X_test, y_test):
    # Predict the classes of test images
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    return predicted_classes, true_classes


def visualise_predictions(X_test, predicted_classes, true_classes):
  """
  Visualizes all images from X_test along with predicted and true classes.

  Args:
      X_test: A numpy array containing the test images.
      predicted_classes: A numpy array containing the predicted class labels.
      true_classes: A numpy array containing the true class labels.
  """

  num_images = len(X_test)  # Use the total number of images

  fig, axes = plt.subplots(num_images, 3, figsize=(12, 2 * num_images))

  for i in range(num_images):  # Loop through all indices
    idx = i  # Use the current index (i)

    ax1, ax2, ax3 = axes[i]

    # True Image
    ax1.imshow(X_test[idx].squeeze(), **helper_cmaps(X_test[idx]))
    ax1.set_title(f'True: {"Healthy" if true_classes[idx] == 1 else "HCM"}')
    ax1.axis('off')

    # Predicted Image
    ax2.imshow(X_test[idx].squeeze(), **helper_cmaps(X_test[idx]))
    ax2.set_title(f'Predicted: {"Healthy" if predicted_classes[idx] == 1 else "HCM"}')
    ax2.axis('off')

    # Difference (optional, for debugging)
    difference = predicted_classes[idx] - true_classes[idx]
    ax3.imshow(X_test[idx].squeeze(), **helper_cmaps(X_test[idx]))
    ax3.set_title(f'Diff: {difference}')
    ax3.axis('off')

  plt.tight_layout()
  plt.show()

def extract_square_patch(image, center, patch_size):
    """
    Extracts a square patch from the image.

    Parameters:
    - image: The source image from which to extract the patch.
    - center: (x, y) center coordinates of the square patch.
    - patch_size: Size of the patch (width and height).

    Returns:
    - The extracted square patch as an image.
    """
    x, y = center
    half_size = patch_size // 2
    x1 = max(0, x - half_size)
    y1 = max(0, y - half_size)
    x2 = min(image.shape[1], x + half_size)
    y2 = min(image.shape[0], y + half_size)

    patch_image = image[y1:y2, x1:x2]

    if patch_image.shape[0] != patch_size or patch_image.shape[1] != patch_size:
        patch_image = cv2.resize(patch_image, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

    return patch_image

def insert_patch(target_image, patch_image, center, patch_size):
    """
    Inserts a square patch into the target image.

    Parameters:
    - target_image: The target image where the patch will be inserted.
    - patch_image: The patch image to be inserted.
    - center: (x, y) center coordinates where the patch will be inserted.
    - patch_size: Size of the patch (width and height).

    Returns:
    - The image with the patch inserted.
    """
    x, y = center
    half_size = patch_size // 2
    x1 = max(0, x - half_size)
    y1 = max(0, y - half_size)
    x2 = min(target_image.shape[1], x + half_size)
    y2 = min(target_image.shape[0], y + half_size)

    patch_resized = cv2.resize(patch_image, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

    target_image[y1:y2, x1:x2] = patch_resized
    
    result_image = target_image

    return result_image

def compute_circular_patch_center(radius, angle, center):
    """
    Computes the center of a patch based on a circular region's angle.

    Parameters:
    - radius: Radius of the circular region.
    - angle: Angle at which the patch is to be placed.
    - center: (x, y) center coordinates of the circular region.

    Returns:
    - (x, y) center coordinates of the patch.
    """
    angle_rad = np.deg2rad(angle)
    x = int(center[0] + radius * np.cos(angle_rad))
    y = int(center[1] + radius * np.sin(angle_rad))
    return (x, y)

# Define the cGAN generator
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

# Define the cGAN discriminator
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

#Generate random labels
def generate_random_labels(num_samples):
    """
    Generates random labels for the fake image generation. Labels are either [0,1] or [1,0].

    Parameters:
    - num_samples: Number of labels to generate

    Returns:
    - Numpy array of random labels
    """
    labels = np.zeros((num_samples, 2))
    random_indices = np.random.randint(0, 2, num_samples)
    labels[np.arange(num_samples), random_indices] = 1
    return labels

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
    return fake_images, labels

'''
Example code

param = 'E2A'
health = 'HCM'
method = 'STEAM'
hs = get_('Healthy',method,'SYSTOLE',param,'myo')
hd = get_('Healthy',method,'DIASTOLE',param,'myo')
uhs = get_('HCM',method,'SYSTOLE',param,'myo')
uhd = get_('HCM',method,'DIASTOLE',param,'myo')
hs = pp(hs)
hd = pp(hd)
uhs = pp(uhs)
uhd = pp(uhd)

h_imgs = hd[0] #retrieve only images not maps
uh_imgs = uhd[0] #retrieve only images not maps


# Example usage
source_images = uh_imgs[:5]
target_images  = uh_imgs[-5:]

for i in range(len(source_images)):
    source_image = source_images[i]
    target_image = target_images[i]
    
    # Display images
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(source_image, **helper_cmaps(source_image))
    plt.title('Source Image',fontsize=7)
    
    plt.subplot(1, 4, 2)
    plt.imshow(target_image, **helper_cmaps(target_image))
    plt.title('Target Image', fontsize=7)
    
    center = pick_centre_middle_(source_image)  # Center of the circular region (x, y)
    radius = center[0]  # Radius of the circular region
    angle_start = 180  # Starting angle (in degrees)
    angle_end = 360  # Ending angle (in degrees)
    patch_size = 40  # Size of the square patch
    
    # Extract patch from source image
    patch_center = compute_circular_patch_center(radius, angle_start, center)
    patch_image = extract_square_patch(source_image, patch_center, patch_size)
    
    # Insert patch into target image
    target_patch_center = compute_circular_patch_center(radius, angle_start, center)
    result_image = insert_patch(target_image, patch_image, target_patch_center, patch_size)
    
    plt.subplot(1, 4, 3)
    plt.imshow(result_image, **helper_cmaps(result_image))
    plt.title('Result Image with Patch', fontsize=7)
    
    plt.subplot(1, 4,4)
    plt.imshow(patch_image, **helper_cmaps(patch_image))
    plt.title('Patch', fontsize=7)
    
    plt.show()
    
    uh_imgs.append(result_image)

X_train,X_test,y_train,y_test = gen_data_split(h_imgs,uh_imgs,6,5) #create data split, taking 5 from healthy and 5 from unhealthy
X_train,X_test = normalise(X_train,X_test)
# X_train,X_test = standardise(X_train,X_test)
X_train,X_test = resize_images(X_train,X_test)

data = X_train, y_train, X_test, y_test

patience = 50
start_from = 100
epochs = 2000

model, callback = build_cnn(patience,start_from)
model.summary()
history,info = train_cnn(model,callback,data,epochs)
plot_history(history)

new_model = model
model_weights = new_model.get_weights()
convolution_weights = model_weights[0].squeeze()
fig, ax = plt.subplots()
im = ax.imshow(convolution_weights,**helper_cmaps([convolution_weights]))
cbar = fig.colorbar(im,shrink=0.5,orientation='horizontal')
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Kernel Weights',size=14)
ax.set_axis_off()
ax.tick_params(axis='both', which='major', labelsize=12)
plt.show()

kernel_weights = new_model.get_weights()
conv_layer = tf.keras.layers.Conv2D(1,5,activation='sigmoid')
input_shape = tf.TensorShape([None, 64, 64, 1])
conv_layer.build(input_shape)
conv_layer.set_weights(kernel_weights[0:2])
person = 3
person_image = X_train[person]
person_image = person_image.reshape(1,64,64,1)
feature_map = conv_layer(person_image)
feature_map = feature_map.numpy().squeeze()

fig, ax = plt.subplots(figsize=(10,6))
ax.set_axis_off()
im= ax.imshow(feature_map,cmap=helper_cmaps([feature_map])['cmap'],vmin=0,vmax=1)
cbar = fig.colorbar(im,shrink=0.5,orientation='horizontal',ticks=[0,0.5,1])
cbar.set_label('Convolutional Layer Activation',size=14)
cbar.ax.tick_params(labelsize=12)
plt.show()

# Predict and visualize
predicted_classes, true_classes= predict_and_visualise(model, X_test, y_test)
visualise_predictions(X_test, predicted_classes, true_classes)

# For the cGAN images
data = X_train
labels = y_train

# Choose the number of images to display
num_images_to_display = 10

# Train the cGAN
history = train_cgan(generator, discriminator, cgan, data, labels, epochs=100, batch_size=32)

X_test_fake,y_test_fake = generate_fake_images(generator, num_images_to_display)

# Predict on the test set
predicted_classes2, true_classes2 = predict_and_visualise(model, X_test_fake, y_test_fake)

# Visualise a subset of test images with predictions
visualise_predictions(X_test_fake, predicted_classes2, true_classes2)
'''


