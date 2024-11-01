import matplotlib.pyplot as plt
import time
import psutil
from dataset import *
from model import *
from dotenv import load_dotenv
import os
import tensorflow as tf

def runWithoutFL():

    load_dotenv()

    epochs = int(os.getenv('EPOCHS'))
    img_path = str(os.getenv('DATASET_PATH'))

    withoutFL = generateModel()
    start_time = time.time()

    (X_train, X_test, y_train, y_test) = getDataSet(img_path)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=1e-6)

    withoutFL.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    history = withoutFL.fit(X_train, y_train, epochs=epochs, batch_size=32)

    end_time = time.time()
    execution_time = end_time - start_time

    ram_usage = psutil.Process().memory_info().rss / 1024 ** 3
    trainable_params = withoutFL.count_params()
    test_loss, test_accuracy = withoutFL.evaluate(X_test, y_test)

    print("RAM usage: {:.2f} GB".format(ram_usage))
    print("Execution time: {:.2f} seconds".format(execution_time))
    print("Trainable parameters: {}".format(trainable_params))
    print("Test accuracy: {:.2f}%".format(test_accuracy * 100))
    print("Loss: {:.2f}%".format(test_loss * 100))

    withoutFL.save("noFL.h5", save_format="h5")
    print("Model noFL saved")

    # Plotting
    acc_color = 'black'
    loss_color = 'darkgray'
    acc_linewidth = 3
    loss_linewidth = 3
    acc_linestyle = '-'
    loss_linestyle = '--'

    epochs_range = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(14, 5))
    plt.plot(epochs_range, history.history['accuracy'], color=acc_color, linestyle=acc_linestyle, linewidth=acc_linewidth)
    plt.plot(epochs_range, history.history['loss'], color=loss_color, linestyle=loss_linestyle, linewidth=loss_linewidth)
    plt.title('Model Performance without Federated Learning', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy/Loss', fontsize=14, fontweight='bold')
    plt.legend(['Accuracy', 'Loss'], fontsize=14, title_fontsize='14', title='Legend').get_title().set_fontweight('bold')

    # Set tick parameters
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('modelNoFL.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('modelNoFL.png', dpi=300, bbox_inches='tight')
    # plt.show()

runWithoutFL()