'''
Code adapted from Tensorflow/Keras documentation.

Generates a random sample of gradcams of size n. 
'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
# Display
import pandas as pd
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from classification_models.tfkeras import Classifiers
import sys
sys.path.append('/home/zzaiman/local/CS334FinalProject/zach')
from trainlib.train import get_model
import os

# Gets the model used in training without parralelization.
def getModel():
    model_init, preprocessing_func = Classifiers.get('densenet121')
    base_model = model_init(include_top=False, input_shape=(224,224,3), weights='imagenet', pooling='max')
    # base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='max')
    x = base_model.output
    x = keras.layers.Dense(512, activation = 'relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    # Add an individual classification layer for every class.
    output = []
    for i in range(14):
        layer_name = "".join(labels[i].split(" ")).lower()
        output.append(keras.layers.Dense(1, activation='sigmoid', name=layer_name)(x))

    model = keras.models.Model(inputs=base_model.input, outputs=output)
    # Prints by default
    model.summary()
    return model, preprocessing_func

# Loads an image. 
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

# Makes the heatmap.
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Overlays heatmap with original image. 


def save_and_display_gradcam(img_path, heatmap, ax, alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    ax.imshow(superimposed_img)
    # superimposed_img.save(cam_path)

# Modifies model object in place to prep for gradcams.
def prep_model(model, weights_path, num_labels):
    # Load weights
    model.load_weights(weights_path)
    prepped_model = tf.keras.models.clone_model(model)
    # Turn off activation function for grad cams.
    for i in range(1, num_labels+1):
        model.layers[-i].activation = None
    
    return prepped_model

def generate_multilabel_gradcam(prepped_model, preprocess, img_path, img_size, dest_path, last_conv_layer, preds, gt):
    fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(30,30))
    img_array = preprocess(get_img_array(img_path, size=img_size))
    
    for i, lab in enumerate(labels):
        
        row = int(i >= 7)
        col = i - (7*row)
        heatmap = make_gradcam_heatmap(img_array, prepped_model, last_conv_layer, pred_index=i)
        save_and_display_gradcam(img_path, heatmap, axes[row, col])
        axes[row, col].set_title(f'{lab}, p={preds[lab]}, gt={gt[lab]}', fontsize=10)
    
    plt.savefig(dest_path, bbox_inches="tight")    

if __name__ == '__main__':

    config = json.load(open('config.json', 'r'))

    # Load necessary variables from config file.
    weights_path = config['WEIGHTS_PATH']
    results_csv_path = config['RESULTS_CSV_PATH']
    image_root_path = config['IMAGE_ROOT_PATH']
    dest_dir = config['DEST_DIR']
    last_layer_name = config['LAST_LAYER_NAME']
    img_height = config['IMG_HEIGHT']
    img_width = config['IMG_WIDTH']
    num_gradcams = config['NUM_GRADCAMS']

    # Create output directory if it doesn't exist.
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Declare the labels.
    global labels 
    labels = 'Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,Pneumothorax,Support Devices'.split(',')
    
    # Get the model architecture.
    model, preprocess = getModel()
    prepped_model = prep_model(model, weights_path, len(labels))

    # Prepare the test CSV
    test = pd.read_csv(results_csv_path)
    test['path'] = test['path'].apply(lambda x: os.path.join(image_root_path, x))
    dest_path = './gradcams/test_gradcam.png'

    img_size = (img_height, img_width)

    # Sample the gradcams according to the user input.
    random_sample = test.sample(n=num_gradcams)

    # Create a gradcam with associated predictions and labels for each image in the sample.
    for i, row in random_sample.iterrows():
        img_path = row['path']
        dest_path = os.path.join(dest_dir, (str(i)) + '.png')
        preds = dict(zip(labels, [row[x] for x in list(test) if 'pred_' in x]))
        gt = dict(zip(labels, [row[x] for x in labels]))
        generate_multilabel_gradcam(prepped_model, preprocess, img_path, img_size, dest_path, last_layer_name, preds, gt)



