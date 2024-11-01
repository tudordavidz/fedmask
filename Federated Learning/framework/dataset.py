import tensorflow as tf
from imutils import paths
import os
from tensorflow.keras.preprocessing.image import load_img # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
import numpy as np
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)



def getDataSet(PathToDataset):

    imagePaths = list(paths.list_images(PathToDataset))

    data = []
    labels = []

    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        # load the input image (224x224) and preprocess it
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)



        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    # convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (X_train, X_test, y_train, y_test) = train_test_split(data, labels,
        test_size=0.20, stratify=labels, random_state=42)

    return (X_train, X_test, y_train, y_test)
