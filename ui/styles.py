"""Futuristic Holographic Sci-Fi Stylesheet for Jarvis PyQt6 Interface.
"""

HOLOGRAPHIC_STYLE = """
QMainWindow, QWidget {
    background-color: #070a10;
    color: #b0e5ff;
    font-family: 'Consolas', 'Segoe UI', 'Roboto', monospace;
    font-size: 13px;
}

/* Glassmorphism holographic panel */
QFrame.holo-panel {
    background-color: rgba(11, 20, 36, 0.75);
    border: 1px solid rgba(0, 240, 255, 0.35);
    border-radius: 8px;
    padding: 8px;
}

QFrame.holo-panel:hover {
    border: 1px solid rgba(0, 240, 255, 0.7);
}

/* Emergency panel highlight */
QFrame.alert-panel {
    background-color: rgba(45, 12, 22, 0.75);
    border: 1px solid rgba(255, 60, 90, 0.6);
    border-radius: 8px;
}

QLabel {
    color: #a0d8ef;
}

QLabel.title {
    font-size: 16px;
    font-weight: bold;
    color: #00f0ff;
    letter-spacing: 2px;
}

QLabel.header {
    font-size: 20px;
    font-weight: bold;
    color: #00f0ff;
    letter-spacing: 3px;
}

QLabel.metric-value {
    font-size: 18px;
    font-weight: bold;
    color: #ffffff;
}

QLabel.chip {
    background-color: rgba(0, 240, 255, 0.15);
    border: 1px solid #00f0ff;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    color: #00f0ff;
    font-weight: bold;
}

QLabel.chip-active {
    background-color: rgba(0, 255, 136, 0.2);
    border: 1px solid #00ff88;
    color: #00ff88;
}

QLabel.chip-alert {
    background-color: rgba(255, 50, 80, 0.2);
    border: 1px solid #ff3250;
    color: #ff3250;
}

QTextEdit, QPlainTextEdit {
    background-color: rgba(6, 12, 22, 0.85);
    border: 1px solid rgba(0, 240, 255, 0.25);
    border-radius: 6px;
    color: #c9f0ff;
    selection-background-color: #0077ff;
    font-family: 'Consolas', monospace;
    font-size: 12px;
}

QLineEdit {
    background-color: rgba(8, 16, 28, 0.9);
    border: 1px solid rgba(0, 240, 255, 0.4);
    border-radius: 5px;
    padding: 6px 10px;
    color: #00f0ff;
    font-family: 'Consolas', monospace;
    font-size: 13px;
}

QLineEdit:focus {
    border: 1px solid #00f0ff;
    background-color: rgba(10, 22, 38, 0.95);
}

QPushButton {
    background-color: rgba(0, 80, 160, 0.4);
    border: 1px solid #00aaff;
    border-radius: 5px;
    color: #00f0ff;
    padding: 6px 14px;
    font-weight: bold;
    letter-spacing: 1px;
}

QPushButton:hover {
    background-color: rgba(0, 140, 255, 0.6);
    border: 1px solid #00ffff;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: rgba(0, 200, 255, 0.8);
    color: #000000;
}

QPushButton.emergency-btn {
    background-color: rgba(160, 20, 40, 0.6);
    border: 1px solid #ff3050;
    color: #ffc0c8;
}

QPushButton.emergency-btn:hover {
    background-color: rgba(220, 20, 50, 0.8);
    border: 1px solid #ff6080;
    color: #ffffff;
}

QProgressBar {
    border: 1px solid rgba(0, 240, 255, 0.3);
    border-radius: 4px;
    background-color: rgba(5, 10, 18, 0.8);
    text-align: center;
    color: #ffffff;
    font-size: 11px;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0077ff, stop:1 #00f0ff);
    border-radius: 3px;
}

QScrollBar:vertical {
    border: none;
    background: rgba(5, 10, 18, 0.5);
    width: 6px;
    border-radius: 3px;
}

QScrollBar::handle:vertical {
    background: rgba(0, 240, 255, 0.3);
    border-radius: 3px;
}

QScrollBar::handle:vertical:hover {
    background: rgba(0, 240, 255, 0.7);
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""
