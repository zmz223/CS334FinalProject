import copy
import sys
import math
from datetime import datetime
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from tensorflow.keras.layers import concatenate, add, GlobalAveragePooling2D, BatchNormalization, Input, Dense, Activation, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import optimizers
import pickle
import os
import json
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
from classification_models.tfkeras import Classifiers

# Returns configuration in dictionary format.
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

def modify_paths(base_path, relative_path):
    return os.path.join(base_path, relative_path)

# Returns DataFrame objects corresponding to train, validation, and test sets with path modified
# relative to the image paths. This function assumes the respective csvs are called train.csv, valid.csv,
# and test.csv. It also assumes there is a path variable with the relative path to the images.
def load_datasets(config):
    csv_base = config['META_BASE_PATH']
    image_base = config['IMAGE_BASE_PATH']
    train = pd.read_csv(os.path.join(csv_base, 'train.csv'))
    valid = pd.read_csv(os.path.join(csv_base, 'valid.csv'))
    test = pd.read_csv(os.path.join(csv_base, 'test.csv'))
    train['path'] = train['path'].apply(lambda x: modify_paths(image_base, x))
    valid['path'] = valid['path'].apply(lambda x: modify_paths(image_base, x))
    test['path'] = test['path'].apply(lambda x: modify_paths(image_base, x))

    return train, valid, test

# Yields labels in required format.
def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        yield (batch_x,[batch_y[:,i] for i in range(14)])


# Returns image data generators for train, valid, and test splits.
def get_image_generators(train, valid, test, config, preprocessing_func):

    HEIGHT = int(config['IMG_HEIGHT'])
    WIDTH = int(config['IMG_WIDTH'])
    BATCH_SIZE = int(config['BATCH_SIZE'])
    TEST_BATCH = int(config['TEST_BATCH'])
    num_classes = int(config['NUM_CLASSES'])

    labels = 'Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,Pneumothorax,Support Devices'.split(',')
    labels = np.array(labels)
    print(f"Num labels {str(len(labels))}", flush=True)

    train_gen = ImageDataGenerator(
            rotation_range=15,
            fill_mode='constant',
            zoom_range=0.1,
            horizontal_flip=True,
            preprocessing_function=preprocessing_func
    )

    validate_gen = ImageDataGenerator(preprocessing_function=preprocessing_func)

    train_batches = train_gen.flow_from_dataframe(
        train,
        directory=None,
        x_col="path",
        y_col=labels,
        class_mode="raw",
        target_size=(HEIGHT, WIDTH),
        shuffle=True,
        seed=1,
        batch_size=BATCH_SIZE
    )

    validate_batches = validate_gen.flow_from_dataframe(
        valid,
        directory=None,
        x_col="path",
        y_col=labels,
        class_mode="raw",
        target_size=(HEIGHT, WIDTH),
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    test_batches = validate_gen.flow_from_dataframe(
        test,
        directory=None,
        x_col="path",
        y_col=labels,
        class_mode=None,
        target_size=(HEIGHT, WIDTH),
        shuffle=False,
        batch_size=TEST_BATCH
    )

    return train_batches, validate_batches, test_batches

def get_model(config):

    num_classes = config['NUM_CLASSES']
    lr = config['INITIAL_LR']
    model_arch = config['MODEL_ARCHITECTURE']
    height = config['IMG_HEIGHT']
    width = config['IMG_WIDTH']

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model_init, preprocessing_func = Classifiers.get(model_arch)
        base_model = model_init(include_top=False, input_shape=(height,width,3), weights='imagenet', pooling='max')
        # base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='max')
        x = base_model.output
        end = Dense(512, activation = 'relu')(x)
        
        # Add an individual classification layer for every class.
        output = []
        for i in range(num_classes):
            x = Dropout(0.3)(end)
            output.append(Dense(1, activation='sigmoid')(x))
        
        model = Model(inputs=base_model.input, outputs=output)
        print(model.summary())

        losses = ['binary_crossentropy' for i in range(num_classes)]
        # Need to compile model with an individual loss function for each layer.
        model.compile(optimizer=Adam(lr),
            loss=losses,
            metrics=[
                tf.keras.metrics.AUC(multi_label=True)
            ]
        )

    return model, preprocessing_func

def train_model(model, train_ds, valid_ds, config):

    BATCH_SIZE = config['BATCH_SIZE']
    TEST_BATCH = config['TEST_BATCH']
    lr = config['INITIAL_LR']
    epochs = config['MAX_EPOCHS']
    output_path = config['OUTPUT_DIR']

    train_epoch = math.ceil(len(train) / BATCH_SIZE)
    val_epoch = math.ceil(len(valid) / BATCH_SIZE)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1,
                                patience=2, min_lr=1e-6, verbose=1)
    ES = EarlyStopping(monitor='val_loss', mode='min', patience=4, restore_best_weights=True)

    weights_dir = os.path.join(output_path, 'weights/')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    weights_path = os.path.join(weights_dir, 'best_model.hdf5')

    checkloss = ModelCheckpoint(weights_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=False)

    history = model.fit(generator_wrapper(train_ds),
        validation_data=generator_wrapper(valid_ds),
        steps_per_epoch=train_epoch,
        validation_steps=val_epoch,
        epochs=epochs,
        shuffle=True,
        callbacks=[reduce_lr, checkloss, ES]
    )

    return history

if __name__ == '__main__':

    config = load_config('config.json')
    train, valid, test = load_datasets(config)

    model, preprocessing_func = get_model(config)

    train_batches, valid_batches, test_batches = get_image_generators(train, valid, test, config, preprocessing_func)

    history = train_model(model, train_batches, valid_batches, config)

    # Predict on test set.
    Y_pred = model.predict(test_batches)

    output_path = config['OUTPUT_DIR']
    # Save pickle files of training history and predictions for further analysis.
    with open(os.path.join(output_path, 'predictions.pkl'), 'wb') as f:
        pickle.dump(Y_pred, f)

    with open(os.path.join('train_hist.pkl'), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

