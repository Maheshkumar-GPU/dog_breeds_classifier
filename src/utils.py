import matplotlib.pyplot as plt
from config import results_path

def plot_results(history):

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")

    plt.legend(["Train", "Validation"])

    plt.savefig(results_path)