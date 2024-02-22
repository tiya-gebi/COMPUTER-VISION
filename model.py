from google.colab import drive
drive.mount('/content/drive', force_remount=True)

#accessing the dataset
DataDir = '/content/drive/MyDrive/FashionItems/data/RingFIR'
class_names = ['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016',
               '017','018','019','020','021','022','023','024','025','026','027','028','029','030','031','032',
               '033','034','035','036','037','038','039','040','041','042','043','044','045','046']

#importing the necessary libraries
import matplotlib.pyplot as plt
import os
import cv2
import seaborn as sns
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.style.use('ggplot')
from sklearn.metrics import classification_report, confusion_matrix

pip install split-folders

#spliting the dataset into 80% of training set, 10% of testing set and 10% of validation set
import splitfolders
input_folder = '/content/drive/MyDrive/FashionItems/data/RingFIR'
splitfolders.ratio(input_folder, output="dataset", seed= 2615, ratio=(0.8, 0.1, 0.1))

test_path = '/content/dataset/test'
train_path = '/content/dataset/train'
val_path = '/content/dataset/val'

#define the shape of the images that we want to train on.
#the images will be resized accordingly before being fed into the neural network
#And  the images resized to 224×224 size
IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = '/content/dataset/train'
TEST_DATA_DIR = '/content/dataset/test'
VALID_DATA_DIR = '/content/dataset/val'

#ImageDataGenerator class can handle the preprocessing of the images
#normalizing the images to scale the pixels between 0 and 1


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.5, 1.5))

train_generator = train_datagen.flow_from_directory(
    TRAINING_DATA_DIR,
    shuffle=True,
    target_size=IMAGE_SHAPE,
)
test_generator = datagen.flow_from_directory(
    TEST_DATA_DIR,
    shuffle=False,
    target_size=IMAGE_SHAPE,
)
valid_generator = datagen.flow_from_directory(
    VALID_DATA_DIR,
    shuffle=False,
    target_size=IMAGE_SHAPE,
)

y_train=train_generator.classes
y_test=test_generator.classes
y_val=valid_generator.classes

#build a simple neural network model consisting of 2D Convolutional layers, 2D Max-Pooling layers, and Linear layers
#and Sequential class to build the model
def build_model(num_classes):
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(filters=128 , kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'), # linear 1d
    tf.keras.layers.Dense(num_classes, activation='softmax') # output
    ])
    return model
model = build_model(num_classes=46)

#to compile the model I Use the Adam optimizer with a learning rate of 0.0001.
#and for the loss function, I use the Categorical Cross-Entropy loss. And the evaluation metric is accuracy.
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
print(model.summary())

#I am training for 50 epochs and the batch size is 32
# and the train_generator that will sample the training data and feed into the neural network.
#The validation_data and the validation_steps is the number of validation samples divided by the batch size.
#I provide verbose=1 to show the progress bar while training.
from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
EPOCHS = 500
BATCH_SIZE = 32
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=valid_generator,
                    validation_steps= valid_generator.samples // BATCH_SIZE,
                    verbose=1,
                    callbacks=[early_stop],
                    shuffle=True
                    )

#code to plot and save the accuracy and loss graphs for both training and validation.
train_loss = history.history['loss']
train_acc = history.history['accuracy']
valid_loss = history.history['val_loss']
valid_acc = history.history['val_accuracy']
def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()
    # loss plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()
save_plots(train_acc, valid_acc, train_loss, valid_loss)

y_pred = model.predict(test_generator)

#evaluate the model
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(20, 15))
y_pred_labels = [np.argmax(label) for label in y_pred ]
cm = confusion_matrix(y_test, y_pred_labels)

#show cm
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)

from sklearn.metrics import classification_report
cr= classification_report (y_test, y_pred_labels, target_names=class_names)
print(cr)

