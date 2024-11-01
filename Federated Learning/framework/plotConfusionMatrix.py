import numpy as np
import matplotlib.pyplot as plt


def plotConfusionMatrix(mean_confusion_matrix, get_num_classes):

    plt.figure()
    # Set font properties for the plot
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'




    # Plot mean confusion matrix
    plt.imshow(mean_confusion_matrix, interpolation='nearest', cmap=plt.cm.Greys)  # Changed to Greys colormap
    plt.title('Mean Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(get_num_classes)
    plt.xticks(tick_marks, tick_marks, weight='bold', fontsize=14)
    plt.yticks(tick_marks, tick_marks, weight='bold', fontsize=14)

    # Adding annotations
    threshold = np.max(mean_confusion_matrix) * 0.75  # Adjust threshold as needed for better clarity
    for i in range(mean_confusion_matrix.shape[0]):
        for j in range(mean_confusion_matrix.shape[1]):
            plt.text(j, i, format(mean_confusion_matrix[i, j], '.2f'),
                    horizontalalignment="center",
                    color="white" if mean_confusion_matrix[i, j] > threshold else "black",
                    weight='bold', fontsize=14)

    plt.tight_layout()
    plt.ylabel('True label', weight='bold', fontsize=14)
    plt.xlabel('Predicted label (c)', weight='bold', fontsize=14)

    # Save as PNG
    plt.savefig("confusion_matrix_colorblind.png", format="png", bbox_inches='tight', dpi=300)
    # Save as PDF
    plt.savefig("confusion_matrix_colorblind.pdf", format="pdf", bbox_inches='tight')

    # plt.show()
