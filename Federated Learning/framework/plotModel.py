import matplotlib.pyplot as plt
import numpy as np

def plotModel(global_acc_list, global_loss_list, comms_round, test_accuracy, test_loss, trainable_params, ram_usage, execution_time):
    plt.figure()    
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    acc_color = 'black'
    loss_color = 'darkgray'
    acc_linestyle = '-'
    loss_linestyle = '--'
    acc_linewidth = 3
    loss_linewidth = 3

    plt.figure(figsize=(14, 5))
    plt.title('Global Model Performance During Federated Learning')
    plt.xlabel('Communication Rounds (a)')
    plt.ylabel('Accuracy/Loss')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.plot(global_acc_list, color=acc_color, linestyle=acc_linestyle, linewidth=acc_linewidth)
    plt.plot(global_loss_list, color=loss_color, linestyle=loss_linestyle, linewidth=loss_linewidth)
    plt.xticks(np.arange(comms_round), np.arange(1, comms_round + 1))

    all_values = global_acc_list + global_loss_list
    plt.yticks(np.linspace(min(all_values), max(all_values), num=10))

    plt.legend(['Accuracy', 'Loss'], loc='best')
    plt.tick_params(axis='both')

    plt.tight_layout()
    plt.savefig("plot_grays.png", dpi=300, bbox_inches='tight')
    plt.savefig("plot_grays.pdf", bbox_inches='tight')

    # plt.show()

    print("RAM usage: {:.2f} GB".format(ram_usage))
    print("Execution time: {:.2f} seconds".format(execution_time))
    print("Trainable parameters: {}".format(trainable_params))
    print("Test accuracy: {:.2f}%".format(test_accuracy * 100))
    print("Loss: {:.2f}%".format(test_loss * 100))

    num_trainable_params = trainable_params
    data_type = np.float32
    bytes_per_param = np.dtype(data_type).itemsize
    size_of_average_weights = num_trainable_params * bytes_per_param
    print("Model size: " + str(size_of_average_weights)) # model size in bytes





