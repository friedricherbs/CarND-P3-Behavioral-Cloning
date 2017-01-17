# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 13:30:05 2016

@author: Friedrich Kenda-Erbs
"""

import numpy as np
import pandas as pd
import os.path as path
import cv2
#from tqdm import tqdm

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

# Fix for bug in keras
import tensorflow
from tensorflow.python.ops import control_flow_ops 
tensorflow.python.control_flow_ops = control_flow_ops

# Define parameters for network
BATCH_SIZE           = 64
NUM_EPOCHS           = 50
STEERING_OFFSET      = 0.15 
MAX_SHIFT            = 25. 
SHIFT_STEERING_ADAPT = 0.2 

# Shift image horizontally
def shift_image(img,shift_x):
    M_shift            = np.float32([[1,0,shift_x],[0,1,0]])
    r,c,channels       = img.shape
    new_img            = cv2.warpAffine(img, M_shift, (c,r))

    return new_img
    
def remove_angles(df, remove_angle, fraction):
    idx_remove       = df[(df['angles']-remove_angle).abs() < 0.01].index
    idx_remove       = np.asarray(idx_remove)
    np.random.shuffle(idx_remove)
    idx_remove       = idx_remove[:int(idx_remove.shape[0]*fraction)]
    df.drop(df.index[idx_remove], inplace=True )
    df               = df.reset_index(drop=True)
    
    return df

def setup_train_data(train):
    
    if train:
        log_path = 'data_old\\driving_log_train.csv'
    else:
        log_path = 'data_old\\driving_log_valid.csv'
            
    df = pd.read_csv(log_path, skipinitialspace=True)
    
    # Center images
    center_data             = df.copy()[['center', 'steering']]
    center_data.columns     = ["images", "angles"]
    center_data['flip']     = np.zeros(df.shape[0], dtype=np.bool_)
    center_data['shift']    = np.zeros(df.shape[0], dtype=np.float)
    center_data['angles']   = pd.to_numeric(center_data['angles'])
    
    # Flipped center images
    center_data_flip            = df.copy()[['center', 'steering']]
    center_data_flip.columns    = ["images", "angles"]
    center_data_flip['flip']    = np.ones(df.shape[0], dtype=np.bool_)
    center_data_flip['shift']   = np.zeros(df.shape[0], dtype=np.float)
    center_data_flip['angles']  = pd.to_numeric(center_data_flip['angles'])
    center_data_flip['angles'] *= -1
    
    # Left images - adapt steering angles
    left_data            = df.copy()[['left', 'steering']]
    left_data.columns    = ["images", "angles"]
    left_data['flip']    = np.zeros(df.shape[0], dtype=np.bool_)
    left_data['shift']   = np.zeros(df.shape[0], dtype=np.float)
    left_data['angles']  = pd.to_numeric(left_data['angles'])
    left_data['angles'] += STEERING_OFFSET
    
    # Right images - adapt steering angles
    right_data            = df.copy()[['right', 'steering']]
    right_data.columns    = ["images", "angles"]
    right_data['flip']    = np.zeros(df.shape[0], dtype=np.bool_)
    right_data['shift']   = np.zeros(df.shape[0], dtype=np.float)
    right_data['angles']  = pd.to_numeric(right_data['angles'])
    right_data['angles'] -= STEERING_OFFSET
    
    # Flipped left images
    left_data_flip              = df.copy()[['left', 'steering']]
    left_data_flip.columns      = ["images", "angles"]
    left_data_flip['flip']      = np.ones(df.shape[0], dtype=np.bool_)
    left_data_flip['shift']     = np.zeros(df.shape[0], dtype=np.float)
    left_data_flip['angles']     = pd.to_numeric(left_data_flip.angles)
    left_data_flip['angles']    += STEERING_OFFSET
    left_data_flip['angles']    *= -1
    
    # Right flipped images
    right_data_flip               = df.copy()[['right', 'steering']]
    right_data_flip.columns       = ["images", "angles"]
    right_data_flip['flip']       = np.ones(df.shape[0], dtype=np.bool_)
    right_data_flip['shift']      = np.zeros(df.shape[0], dtype=np.float)
    right_data_flip['angles']     = pd.to_numeric(right_data_flip['angles'])
    right_data_flip['angles']    -= STEERING_OFFSET
    right_data_flip['angles']    *= -1
    
    all_data = pd.concat([center_data, left_data, right_data, center_data_flip, left_data_flip, right_data_flip], ignore_index=True)
    
    # Remove overrepresented values  
    # Remove zero angle steering
    all_data              = remove_angles(all_data, 0.0, 0.9)
    
    # Remove positive steering offsets
    all_data              = remove_angles(all_data, STEERING_OFFSET, 0.9)
    
    # Remove negative steering offsets
    all_data              = remove_angles(all_data, STEERING_OFFSET, -0.9)
    
    # Randomly shift images and adapt steering angles accordingly
    rand_shift          = (-1.0+2.0*np.random.uniform(size=all_data.shape[0]))*MAX_SHIFT
    all_data['shift']   = rand_shift
    all_data['angles'] += rand_shift/MAX_SHIFT*SHIFT_STEERING_ADAPT
    
    return all_data

# Generator for training and validation data
def generate_data_from_file(data):
    
    num_batches  = data.shape[0] // BATCH_SIZE
    assert(num_batches > 0)
    
    img_path     = "data_old\\IMG_RESIZED\\"
    
    x = np.zeros(shape=(BATCH_SIZE,66,200,3))
    y = np.zeros(BATCH_SIZE)
    
    while True:
        
        # Shuffle data once in every epoch
        data         = data.sample(frac=1).reset_index(drop=True)  
        img_files    = data['images']
        
        for batch in range(num_batches):
            
            start_idx = batch*BATCH_SIZE       
            for frame in range(BATCH_SIZE):
                    
                new_idx = start_idx+frame
                assert(new_idx < img_files.shape[0])
                img_name = data.get_value(new_idx, 'images') 
                
                assert(path.exists(img_path + img_name))
                # Read images
                img = cv2.imread(img_path + img_name)
                # Convert color representation
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                
                # Flip images horizontally
                if (data.get_value(new_idx, 'flip') == True):
                    img = cv2.flip(img,1)
                    
                # Shift images horizontally
                img = shift_image(img.astype(float), data.get_value(new_idx, 'shift'))
                
                x[frame,:,:,:] = img
                
                steering_angle = data.get_value(new_idx, 'angles')
                assert(np.isinf(steering_angle) == False)
                assert(np.isnan(steering_angle) == False)
                y[frame]  = steering_angle
                
            yield(x, y)

# Define input image dimensions - images have been resized before and stored
ch, input_row, input_col = 3, 66, 200

# Setup neural network
model = Sequential()

model.add(Lambda(lambda x: x/127.5 -1., input_shape=(input_row, input_col, ch)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.summary()
    
# Compile and train the model 
adam = Adam(lr=1e-3)
model.compile(loss='mse',
              optimizer='adam',
              metrics=['mean_squared_error'])

# Define callbacks for training
checkpoint = ModelCheckpoint('model.{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# Setup training and validation data
# Training data has been split beforehand into 80% training data and 20% validation data
print('Setting up train and validation data ...')
train_data = setup_train_data(True)
print('Train data created! Size: {}'.format(train_data.shape[0]))
valid_data = setup_train_data(False)
print('Validation data created! Size: {}'.format(valid_data.shape[0]))

# Show training data
train_data.angles.plot.hist(alpha=0.5, bins=50)

# Define number of training and validation epochs for generator
steps_per_train_epoch = train_data.shape[0] // BATCH_SIZE
steps_per_valid_epoch = valid_data.shape[0] // BATCH_SIZE

history = model.fit_generator( generate_data_from_file(train_data), 
                     samples_per_epoch=steps_per_train_epoch*BATCH_SIZE, 
                     nb_epoch=NUM_EPOCHS, 
                     verbose=1,
                     validation_data=generate_data_from_file(valid_data), 
                     nb_val_samples=steps_per_valid_epoch*BATCH_SIZE,
                     callbacks=[checkpoint, early_stopping] )

# Save architecture as JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
print("Saved model to disk")