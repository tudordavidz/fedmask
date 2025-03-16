from sklearn.model_selection import RepeatedKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K # type: ignore
import gc
import scipy.stats as stats
from scipy import stats

def compute_cv_statistics(metric_list, baseline=0.0, alpha=0.05):
    """
    Compute summary statistics for a list of cross-validation metrics.

    Parameters:
        metric_list (list or np.array): The metric values (e.g., accuracy or loss).
        baseline (float): The value to compare against in the one-sample t-test.
        alpha (float): Significance level for the confidence interval.

    Returns:
        dict: A dictionary containing the mean, standard deviation, confidence interval, and p-value.
    """
    metrics = np.array(metric_list)
    n = len(metrics)
    mean_val = np.mean(metrics)
    std_val = np.std(metrics, ddof=1)
    se = std_val / np.sqrt(n)

    # Calculate the t critical value for two-tailed test
    t_crit = stats.t.ppf(1 - alpha/2, n - 1)
    ci_lower = mean_val - t_crit * se
    ci_upper = mean_val + t_crit * se

    # One-sample t-test comparing metric values to the baseline value.
    t_stat, p_value = stats.ttest_1samp(metrics, baseline)

    return {
        'mean': mean_val,
        'std_dev': std_val,
        'confidence_interval': (ci_lower, ci_upper),
        'p_value': p_value
    }

def print_cv_results_table(val_acc_list, val_loss_list, baseline_acc=0.0, baseline_loss=0.0):
    """
    Print a table of CV results statistics for accuracy and loss.

    Parameters:
        val_acc_list (list): List of validation accuracy values.
        val_loss_list (list): List of validation loss values.
        baseline_acc (float): Baseline accuracy for t-test (e.g. chance level).
        baseline_loss (float): Baseline loss for t-test.
    """
    acc_stats = compute_cv_statistics(val_acc_list, baseline=baseline_acc)
    loss_stats = compute_cv_statistics(val_loss_list, baseline=baseline_loss)
    
    print("------------------------------------")
    print("\n Acc List : ", val_acc_list)
    print("\n Loss List : ", val_loss_list) 
    print("------------------------------------")
    
    header = f"{'Metric':<15} {'Mean':<10} {'Std Dev':<10} {'Confidence Interval':<30} {'p-value':<10}"
    line = "-" * len(header)
    print(header)
    print(line)
    print(f"{'Accuracy':<15} {acc_stats['mean']:<10.4f} {acc_stats['std_dev']:<10.4f} "
          f"[{acc_stats['confidence_interval'][0]:.4f}, {acc_stats['confidence_interval'][1]:.4f}] {acc_stats['p_value']:<10.4f}")
    print(f"{'Loss':<15} {loss_stats['mean']:<10.4f} {loss_stats['std_dev']:<10.4f} "
          f"[{loss_stats['confidence_interval'][0]:.4f}, {loss_stats['confidence_interval'][1]:.4f}] {loss_stats['p_value']:<10.4f}")

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

        # Compute confusion matrix for this fold and add to cumulative sum
        y_pred = model.predict(X_val_fold)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val_fold, axis=1)
        fold_confusion_matrix = confusion_matrix(y_true_classes, y_pred_classes, labels=np.arange(num_classes))
        cumulative_confusion_matrix += fold_confusion_matrix

        count += 1
        K.clear_session()
        gc.collect()

    # Compute mean confusion matrix
    mean_confusion_matrix = cumulative_confusion_matrix / (n_splits * n_repeats)

    # After cross-validation, print the summary table for accuracy and loss.
    # For classification accuracy, you might choose chance level as baseline (e.g., 1/num_classes)
    baseline_acc = 1.0 / num_classes
    print_cv_results_table(val_acc_list, val_loss_list, baseline_acc=baseline_acc, baseline_loss=0.0)

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