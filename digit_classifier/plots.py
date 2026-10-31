"""Save scientific charts without opening a desktop plotting window."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def figure(title, subtitle, size=(10, 6)):
    fig = Figure(figsize=size, dpi=160, facecolor="#f8fafc")
    FigureCanvasAgg(fig)
    fig.text(.1, .94, title, fontsize=19, weight="bold", color="#17324d")
    fig.text(.1, .88, subtitle, fontsize=10, color="#475569")
    fig.text(.1, .04, "PJ11 | MNIST handwritten digits | TensorFlow / Keras", fontsize=9, color="#64748b")
    fig.subplots_adjust(left=.12, right=.94, bottom=.18, top=.78)
    return fig


def save_plots(history, labels, probabilities, images, output):
    fig = figure("Learning progress", "Training and validation accuracy; test set excluded from model selection")
    ax = fig.add_subplot()
    epochs = np.arange(1, len(history["accuracy"]) + 1)
    ax.plot(epochs, history["accuracy"], "o-", label="Training", color="#0284c7")
    ax.plot(epochs, history["val_accuracy"], "o-", label="Validation", color="#e87924")
    ax.set(xlabel="Epoch", ylabel="Accuracy", xticks=epochs, ylim=(0, 1))
    ax.grid(alpha=.2)
    ax.legend(loc="lower right")
    fig.savefig(output / "learning_curve.png")

    predicted = probabilities.argmax(1)
    matrix = np.zeros((10, 10), dtype=int)
    np.add.at(matrix, (labels, predicted), 1)
    fig = figure("Test set confusion matrix", "Rows: actual digit | Columns: predicted digit | Raw image counts", size=(9, 8))
    ax = fig.add_subplot()
    plot = ax.imshow(matrix, cmap="Blues")
    for row in range(10):
        for col in range(10):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=8,
                    color="white" if matrix[row, col] > matrix.max() / 2 else "#17324d")
    ax.set(xlabel="Predicted digit", ylabel="Actual digit", xticks=range(10), yticks=range(10))
    fig.colorbar(plot, ax=ax, label="Images")
    fig.savefig(output / "confusion_matrix.png")

    chosen = np.arange(min(12, len(images)))
    fig = figure("Handwritten digit predictions", "Untouched test samples | True label, predicted digit and model probability", size=(10, 8))
    for panel, idx in enumerate(chosen, 1):
        ax = fig.add_subplot(3, 4, panel)
        ax.imshow(images[idx], cmap="gray", vmin=0, vmax=1)
        color = "#15803d" if labels[idx] == predicted[idx] else "#b91c1c"
        ax.set_title(f"True {labels[idx]} | Pred {predicted[idx]} ({probabilities[idx].max():.0%})", fontsize=9, color=color)
        ax.axis("off")
    fig.subplots_adjust(hspace=.5)
    fig.savefig(output / "sample_predictions.png")
