from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.layers import Input # type: ignore
from tensorflow.keras.layers import AveragePooling2D # type: ignore
from tensorflow.keras.layers import Dropout # type: ignore
from tensorflow.keras.layers import Flatten # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.models import Model # type: ignore
import tensorflow as tf
from sklearn.metrics import accuracy_score

def generateModel():
      baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3))) # weights="imagenet"

      # construct the head of the model that will be placed on top of the
      # the base model
      headModel = baseModel.output
      headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
      headModel = Flatten(name="flatten")(headModel)
      headModel = Dense(128, activation="relu")(headModel)
      headModel = Dropout(0.5)(headModel)
      headModel = Dense(2, activation="softmax")(headModel)

      # place the head FC model on top of the base model (this will become
      # the actual model we will train)
      model = Model(inputs=baseModel.input, outputs=headModel)

      # loop over all layers in the base model and freeze them so they will
      # *not* be updated during the first training process
      for layer in baseModel.layers:
        layer.trainable = False
      return model


def test_model(X_test, Y_test,  model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss