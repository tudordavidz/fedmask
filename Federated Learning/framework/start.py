from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import gc
import time
import psutil
from dataset import *
from clients import *
from model import *
from utils import *
from plotModel import plotModel
from plotRepeatedKFold import plotBoxAndWhisker
from plotConfusionMatrix import plotConfusionMatrix
from dotenv import load_dotenv

load_dotenv()

n_splits = int(os.getenv('NR_SPLITS'))
n_repeats = int(os.getenv('NR_REPEATS'))

def startFL (): 

    num_clients = int(os.getenv('NR_CLIENTS'))
    comms_round = int(os.getenv('COMMS_ROUND'))
    img_path = str(os.getenv('DATASET_PATH'))

    (X_train, X_test, y_train, y_test) = getDataSet(img_path)

    clients = create_clients(X_train, y_train, num_clients, initial='client')

    clients_batched = dict()
    for (client_name, data) in clients.items():
        clients_batched[client_name] = batch_data(data)


    test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=1e-6)

    global_model = generateModel()
    global_acc_list = []
    global_loss_list = []


    start_time = time.time()

    for comm_round in range(comms_round):
        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()

        scaled_local_weight_list = list()

        #randomize client data - using keys
        client_names= list(clients_batched.keys())
        random.shuffle(client_names)

        #loop through each client and create new local model
        for client in client_names:
            print("CLIENT NAME: "+str(client))
            local_model = generateModel()
            local_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
            local_model.set_weights(global_weights)

            #fit local model with client's data
            local_model.fit(clients_batched[client], epochs=1, verbose=0)

            #scale the model weights and add to list
            scaling_factor = weight_scalling_factor(clients_batched, client)
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)

            #clear session to free memory after each communication round
            K.clear_session()
            gc.collect()

        #to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_local_weight_list)

        #update global model
        global_model.set_weights(average_weights)

        #test global model and print out metrics after each communications round
        for(X_test, Y_test) in test_batched:
            global_acc, global_loss = test_model(X_test, y_test, global_model, comm_round)
            global_acc_list.append(global_acc)
            global_loss_list.append(global_loss)


    end_time = time.time()
    execution_time = end_time - start_time
    ram_usage = psutil.Process().memory_info().rss / 1024 ** 3
    trainable_params = global_model.count_params()
    test_loss, test_accuracy = global_loss, global_acc

    global_model.save("FLModel.h5", save_format="h5")
    print ('[STATUS] Federating Model Saved' )

    return global_model, test_accuracy, test_loss, trainable_params, ram_usage, execution_time, global_acc_list, global_loss_list, comms_round, X_train, y_train


#start the federated learning process
global_model, test_accuracy, test_loss, trainable_params, ram_usage, execution_time, global_acc_list, global_loss_list, comms_round,  X_train, y_train = startFL()

#plot the results
plotModel(global_acc_list, global_loss_list, comms_round, test_accuracy, test_loss, trainable_params, ram_usage, execution_time)
mean_confusion_matrix, get_num_classes = plotBoxAndWhisker(global_model, X_train, y_train, n_splits, n_repeats)
plotConfusionMatrix(mean_confusion_matrix, get_num_classes)