import sys
from pathlib import Path

# Ensure jarvis root is in sys.path for test execution
JARVIS_DIR = Path(__file__).resolve().parent
if str(JARVIS_DIR) not in sys.path:
    sys.path.insert(0, str(JARVIS_DIR))
