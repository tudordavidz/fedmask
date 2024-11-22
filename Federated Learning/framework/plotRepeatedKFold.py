from sklearn.model_selection import RepeatedKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K # type: ignore
import gc

def cross_validate(model, X, y, n_splits, n_repeats):
    num_classes = y.shape[1]
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
    val_acc_list = []
    val_loss_list = []

    cumulative_confusion_matrix = np.zeros((num_classes, num_classes))

    count = 1
    for train_index, val_index in rkf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])  # For multi-class classification

        model.fit(X_train_fold, y_train_fold, epochs=1, verbose=0)
        loss, acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        print(f'Iteration: {count}, Accuracy: {acc}, Loss: {loss}')
        val_acc_list.append(acc)
        val_loss_list.append(loss)

        # Get predictions
        y_pred = model.predict(X_val_fold)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val_fold, axis=1)

        # Compute confusion matrix for this fold and add to cumulative sum
        fold_confusion_matrix = confusion_matrix(y_true_classes, y_pred_classes, labels=np.arange(num_classes))
        cumulative_confusion_matrix += fold_confusion_matrix

        count += 1
        K.clear_session()
        gc.collect()

    # Compute mean confusion matrix
    mean_confusion_matrix = cumulative_confusion_matrix / (n_splits * n_repeats)

    return val_acc_list, val_loss_list, mean_confusion_matrix


def startPlotBoxAndWhisker(val_acc_list, val_loss_list, n_splits, n_repeats):
    plt.figure()
    # Set font properties for the plot
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    # reshape accuracy list into a 2D array where each row represents a repeat of k-fold cv
    acc_arr = np.array(val_acc_list).reshape(n_repeats, n_splits)
    loss_arr = np.array(val_loss_list).reshape(n_repeats, n_splits)

    fig, ax1 = plt.subplots(figsize=(14,5))

    pos_acc = np.arange(1, 2*n_repeats, 2)
    pos_loss = np.arange(2, 2*n_repeats + 1, 2)

    # Accuracy plot on primary y-axis
    bp1 = ax1.boxplot(acc_arr, positions=pos_acc, showmeans=True, meanline=False,
                      meanprops=dict(marker='^', markerfacecolor='black', markersize=10),
                      patch_artist=True, boxprops=dict(facecolor='black'))

    ax1.set_title('Box and Whisker Plots of Accuracy and Loss for Each Repeat', color='black')
    ax1.set_xlabel('Repeats (b)', color='black')
    ax1.set_ylabel('Accuracy', color='black')
    ax1.tick_params('y', colors='black')
    ax1.tick_params('x', colors='black')

    # Set tick labels bold
    for label in ax1.get_xticklabels():
        label.set_weight('bold')
        label.set_color('black')

    # Set ytick labels with 4 decimal places for ax1
    ax1.set_yticklabels(['{:.4f}'.format(x) for x in ax1.get_yticks()])

    for label in ax1.get_yticklabels():
        label.set_weight('bold')
        label.set_color('black')

    # Create twin y-axis for the loss
    ax2 = ax1.twinx()
    bp2 = ax2.boxplot(loss_arr, positions=pos_loss, showmeans=True, meanline=False,
                      meanprops=dict(marker='^', markerfacecolor='darkgray', markersize=10),
                      patch_artist=True, boxprops=dict(facecolor='darkgray'))
    ax2.set_ylabel('Loss', color='darkgray')
    ax2.tick_params('y', colors='darkgray')

    # Set ytick labels with 4 decimal places for ax2
    ax2.set_yticklabels(['{:.4f}'.format(x) for x in ax2.get_yticks()])


    for label in ax2.get_yticklabels():
        label.set_weight('bold')
        label.set_color('darkgray')

    # Set x-ticks
    ax1.set_xticks((pos_acc + pos_loss) / 2)
    ax1.set_xticklabels(range(1, n_repeats + 1))

    # Handling of legends
    lines = [bp1["boxes"][0], bp2["boxes"][0]]
    ax1.legend(lines, ['Accuracy', 'Loss'], loc='best')

    plt.tight_layout()

    # Save as PNG
    plt.savefig("boxPlot.png", format="png", bbox_inches='tight', dpi=300)
    # Save as PDF
    plt.savefig("boxPlot.pdf", format="pdf", bbox_inches='tight')

    #plt.show()

def plotBoxAndWhisker(global_model, X_train, y_train, n_splits, n_repeats):
    val_acc_list, val_loss_list, mean_confusion_matrix = cross_validate(global_model, X_train, y_train, n_splits, n_repeats)
    get_num_classes = y_train.shape[1] 
    startPlotBoxAndWhisker(val_acc_list, val_loss_list, n_splits, n_repeats)
    return mean_confusion_matrix, get_num_classes