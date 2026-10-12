"""Neural Model & Dataset Training Panel for Jarvis.
Displays active model metrics, dataset status, and live training progress.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)


class TrainingPanel(QFrame):
    """Visual monitor for PyTorch intent model metrics and training progress."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "holo-panel")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Title
        title = QLabel("NEURAL INTENT NETWORK")
        title.setProperty("class", "title")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(6)

        grid.addWidget(QLabel("Active Version:"), 0, 0)
        self.lbl_version = QLabel("v1")
        self.lbl_version.setProperty("class", "chip")
        grid.addWidget(self.lbl_version, 0, 1)

        grid.addWidget(QLabel("Dataset Size:"), 0, 2)
        self.lbl_dataset_size = QLabel("0 examples")
        grid.addWidget(self.lbl_dataset_size, 0, 3)

        grid.addWidget(QLabel("Accuracy:"), 1, 0)
        self.lbl_accuracy = QLabel("N/A")
        self.lbl_accuracy.setStyleSheet("color: #00ff88; font-weight: bold;")
        grid.addWidget(self.lbl_accuracy, 1, 1)

        grid.addWidget(QLabel("Loss:"), 1, 2)
        self.lbl_loss = QLabel("N/A")
        grid.addWidget(self.lbl_loss, 1, 3)

        layout.addLayout(grid)

        # Training Progress
        self.lbl_progress_status = QLabel("Model Status: Online & Ready")
        layout.addWidget(self.lbl_progress_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setFixedHeight(12)
        layout.addWidget(self.progress_bar)

    def set_metrics(self, version: int, dataset_size: int, accuracy: float, loss: float):
        self.lbl_version.setText(f"v{version}")
        self.lbl_dataset_size.setText(f"{dataset_size} examples")
        self.lbl_accuracy.setText(f"{accuracy * 100:.1f}%" if accuracy > 0 else "Pending")
        self.lbl_loss.setText(f"{loss:.4f}" if loss > 0 else "Pending")

    def update_training_progress(self, epoch: int, total_epochs: int, loss: float, acc: float):
        pct = int((epoch / total_epochs) * 100)
        self.progress_bar.setValue(pct)
        self.lbl_progress_status.setText(
            f"Training: Epoch {epoch}/{total_epochs} | Loss: {loss:.4f} | Acc: {acc * 100:.0f}%"
        )
        if epoch == total_epochs:
            self.lbl_progress_status.setText("Model Retraining Complete.")
