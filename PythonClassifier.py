import os
import cv2
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from moviepy.editor import *
import pandas as pd
 
from sklearn.model_selection import train_test_split
 
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

seed_constant = 23
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)
 
# Get Names of all classes
classes_list = os.listdir('ASLLRP\\train')

image_height, image_width = 64, 64
max_images_per_class = 200
train_dir = f'D:\\Uni\\capstone\\ASLLRP\\train'
test_dir = f'D:\\Uni\\capstone\\ASLLRP\\test'
model_output_size = len(classes_list)

def frames_extraction(video_path):
    # take in video and output array of normalised frames
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)

    while True:
        success, frame = video_reader.read() 
        if not success:
            break

        resized_frame = cv2.resize(frame, (image_height, image_width))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)

    video_reader.release() 
    return frames_list


def create_dataset():
    # extract videos in each class and return features and labels of each video
    temp_features = [] 
    features = []
    labels = []
    
    for class_index, class_name in enumerate(classes_list):
        print(f'Extracting Data of Class: {class_name}')
        files_list = os.listdir(os.path.join(train_dir, class_name))

        for file_name in files_list:
            video_file_path = os.path.join(train_dir, class_name, file_name)
            frames = frames_extraction(video_file_path)
            temp_features.extend(frames)

        features.extend(random.sample(temp_features, 200))
        labels.extend([class_index] * max_images_per_class)
        temp_features.clear()

    features = np.asarray(features)
    labels = np.array(labels)  

    return features, labels

features, labels = create_dataset()
one_hot_encoded_labels = to_categorical(labels)

features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size = 0.15, shuffle = True, random_state = seed_constant)

def create_model():
    # construct convolutional neural network architecture
    model = Sequential()
 
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', input_shape = (image_height, image_width, 3)))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(model_output_size, activation = 'softmax'))
 
    model.summary()
 
    return model
 
model = create_model()
 
print("Model Created Successfully!")

plot_model(model, to_file = 'model_structure_plot.png', show_shapes = True, show_layer_names = True)

# Stop training early if the model's validation loss does not decrease after 15 consecutive epochs
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'min', restore_best_weights = True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

# Start Training
model_training_history = model.fit(x = features_train, y = labels_train, epochs = 50, batch_size = 4 , shuffle = True, validation_split = 0.2, callbacks = [early_stopping_callback])
model_evaluation_history = model.evaluate(features_test, labels_test)

# Save model
date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
model_name = f'Model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'
model.save(model_name)

# # Load model (Above code can be commented out and replaced with below code if a model already exists)

# model = create_model()
# plot_model(model, to_file = 'model_structure_plot.png', show_shapes = True, show_layer_names = True)
# model.load_weights('Model___Date_Time_2022_10_23__20_21_40___Loss_0.7832487225532532___Accuracy_0.7333333492279053.h5')


def make_average_predictions(video_file_path):

    video_reader = cv2.VideoCapture(video_file_path)
    window_size = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    predicted_labels_probabilities_np = np.zeros((window_size, model_output_size), dtype = np.float)

    for frame_counter in range(window_size): 

        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
        _ , frame = video_reader.read() 
        resized_frame = cv2.resize(frame, (image_height, image_width))
        normalized_frame = resized_frame / 255


        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]
        predicted_labels_probabilities_np[frame_counter] = predicted_labels_probabilities
    
    predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)
    predicted_labels_probabilities_averaged_sorted_indexes = np.argsort(predicted_labels_probabilities_averaged)[::-1]

    label = predicted_labels_probabilities_averaged_sorted_indexes[0]
    predicted_class_name = classes_list[label]
    predicted_label_probability = predicted_labels_probabilities_averaged[label]

    video_reader.release()
    return predicted_class_name, predicted_label_probability
    

df = pd.DataFrame(columns=["File", "True Class", "Predicted Class", "Predicted Class Probability"])
for root, _, files in os.walk(test_dir, topdown=False):
    for name in files:
        input_video_file_path = os.path.join(root, name)
        print("\nPredicting file: " + input_video_file_path)
        predicted_class, predicted_class_probability = make_average_predictions(input_video_file_path)
        trueClassName = root.split('\\')[-1]
        df.loc[len(df)] = [input_video_file_path, trueClassName, predicted_class, round(predicted_class_probability, 2)]

df.to_csv('Predictions.csv', index=False)
