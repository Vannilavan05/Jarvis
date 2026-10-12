# JARVIS — Voice-Controlled Deep Learning Desktop Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6%20Holographic-brightgreen.svg)](https://riverbankcomputing.com/software/pyqt/)
[![Tests](https://img.shields.io/badge/tests-28%20passed-success.svg)](tests/)

**Jarvis** is a complete, production-oriented Python desktop assistant inspired by the JARVIS system from Iron Man. It integrates real-time offline voice recognition, a genuine trainable deep-learning neural intent classifier implemented in PyTorch, strict permission-checked task execution, spoken text-to-speech feedback with acoustic loopback prevention, and a futuristic PyQt6 holographic interface.

---

## 1. Core Architecture & Design Principles

Jarvis operates under a **Strict User-Command Execution Policy**:
- **Listening & Autonomous Boundary**: Jarvis listens and acts **only** upon explicit, authorized user commands. It never initiates tasks autonomously, never infers permission, and never executes actions merely because the AI model predicted an intent.
- **Unified Command Lifecycle**:
  $$\text{Wake Word} \longrightarrow \text{Speech Recognition} \longrightarrow \text{Intent Prediction} \longrightarrow \text{Parameter Validation} \longrightarrow \text{Authorization Check} \longrightarrow \text{Confirmation (if required)} \longrightarrow \text{Execution} \longrightarrow \text{Verification} \longrightarrow \text{Spoken Feedback} \longrightarrow \text{Audit Logging}$$
- **Voice-First Operation**: Every normal setting, training step, model evaluation, rollback, and execution can be controlled entirely through spoken instructions. The GUI serves as a visual holographic dashboard with optional manual fallbacks.

```
jarvis/
├── main.py                     # Central system bootstrap & CLI/GUI launcher
├── requirements.txt            # Pinned dependency definitions
├── jarvis.spec                 # PyInstaller standalone packaging specification
├── conftest.py                 # Pytest configuration and sys.path setup
├── config/
│   ├── __init__.py             # Directory path constants
│   └── settings.json           # JSON persistent configuration
├── voice/
│   ├── audio_capture.py        # sounddevice microphone capture, RMS amplitude calculation
│   ├── wake_word.py            # Local offline wake-word detector with alias support
│   ├── speech_recognizer.py    # Offline Vosk speech-to-text with error resilience
│   ├── speech_output.py        # pyttsx3 offline TTS with loopback suppression
│   └── audio_manager.py        # Central state machine (IDLE, LISTENING, PROCESSING, SPEAKING)
├── ai/
│   ├── model.py                # PyTorch JarvisIntentNet bidirectional GRU + classifier
│   ├── tokenizer.py            # Vocabulary builder, text-to-tensor encoder, sequence padding
│   ├── dataset.py              # PyTorch Dataset and train/val/test splitter
│   ├── trainer.py              # PyTorch training loop with Adam optimizer and CrossEntropyLoss
│   ├── evaluator.py            # Accuracy, loss, per-intent metrics, confidence calibration
│   ├── predictor.py            # Inference engine with softmax confidence estimation
│   └── model_registry.py       # Checkpoint persistence (v1, v2...), metadata, and rollback
├── nlp/
│   ├── intent_parser.py        # Emergency rules + PyTorch prediction + validation
│   ├── entity_extractor.py     # Parameter extraction (applications, files, search, metrics)
│   └── confidence.py           # Confidence thresholding & clarification generation
├── security/
│   ├── permissions.py          # PermissionLevel hierarchy (READ_ONLY to PRIVILEGED)
│   ├── validator.py            # Path traversal defense, directory sandboxing, argument checks
│   ├── confirmation_manager.py # Two-phase spoken confirmation with timeout expiration
│   ├── authentication.py       # Biometric speaker verification hook
│   └── audit_logger.py         # Local SQLite audit records for all operations
├── execution/
│   ├── command_registry.py     # Registry of approved intents, params, and permissions
│   ├── plugin_manager.py       # Plugin discovery, allowlisting, and intent routing
│   └── dispatcher.py           # Central authorization and execution pipeline
├── plugins/
│   ├── base.py                 # BasePlugin abstract interface
│   ├── applications.py         # Safe launching & termination of approved apps
│   ├── files.py                # Sandboxed file reading, creation, and deletion
│   ├── web_search.py           # URL-encoded web queries via default browser
│   ├── system_info.py          # psutil hardware statistics (CPU, RAM, Disk, Battery, OS)
│   ├── system_controls.py      # Volume control, screen lock, power management
│   ├── model_management.py     # Spoken dataset additions, training, evaluation, rollback
│   └── custom_commands.py      # User-registered voice macros
├── storage/
│   ├── database.py             # SQLite thread-safe manager with WAL mode
│   ├── settings_store.py       # Atomic settings persistence
│   └── training_store.py       # Dataset storage, versioning, and baseline seeding
├── ui/
│   ├── styles.py               # Sci-Fi holographic dark cyan stylesheet
│   ├── waveform.py             # Holographic Arc Reactor widget with rotating tech rings
│   ├── dashboard.py            # Real-time telemetry grid and status chips
│   ├── training_panel.py       # Neural network metrics and live training progress bar
│   └── main_window.py          # Master PyQt6 responsive desktop window
├── tests/                      # Comprehensive pytest automated test suite (28 tests)
│   ├── test_voice.py           # Wake words, audio devices, TTS loopback suppression
│   ├── test_model.py           # PyTorch forward pass, tokenizer, padding
│   ├── test_training.py        # Dataset splitting, training loop, rollback
│   ├── test_security.py        # Path traversal rejection, confirmation expiration, audit log
│   ├── test_permissions.py     # Permission rank hierarchy & unauthorized rejection
│   ├── test_execution.py       # Dispatcher lifecycle, emergency stop
│   ├── test_plugins.py         # App launcher, file sandbox, search, system info
│   └── test_conversational_workflow.py # Spoken training flow & injection defense
├── models/                     # Saved PyTorch model checkpoints & Vosk STT models
├── data/                       # Local SQLite database (`jarvis.db`)
└── logs/                       # Application logs (`jarvis.log`)
```

---

## 2. Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| **Runtime** | Python 3.11+ (verified up to 3.14) | Modern async & typing support |
| **GUI Framework** | PyQt6 | Hardware-accelerated holographic rendering |
| **Offline Speech STT** | Vosk | High-accuracy local Kaldi speech-to-text |
| **Deep Learning** | PyTorch | Trainable neural text classifier with recurrent layer |
| **Text-to-Speech (TTS)** | pyttsx3 (SAPI5/eSpeak) | Fast local TTS with speaking-state mute protection |
| **Audio Capture** | sounddevice & numpy | Low-latency audio stream and RMS computation |
| **Local Database** | SQLite (WAL mode) | ACID-compliant storage for dataset & audit logs |
| **Testing** | pytest | Automated unit and integration testing |
| **Packaging** | PyInstaller | Standalone single-directory or single-file desktop build |

---

## 3. Voice-Only Control Command Reference

Every capability in Jarvis can be operated by voice without touching a keyboard or mouse.

### Wake Word & Listening
- `"Jarvis"` (Default wake word)
- `"Hey Jarvis"` / `"Okay Jarvis"`
- `"Jarvis, pause listening"` / `"Jarvis, pause"`
- `"Jarvis, resume listening"`
- `"Jarvis, stop listening"` (Emergency command)
- `"Jarvis, cancel the current command"` / `"Jarvis, return to idle mode"`

### Applications
- `"Jarvis, open Chrome"` / `"Launch browser"`
- `"Jarvis, open Notepad"` / `"Start text editor"`
- `"Jarvis, open Calculator"`
- `"Jarvis, open File Explorer"`
- `"Jarvis, open VS Code"`
- `"Jarvis, close Chrome"` *(Requires spoken confirmation)*

### File Operations (Strictly Sandboxed)
- `"Jarvis, list files"` / `"Show files in directory"`
- `"Jarvis, read file notes.txt"`
- `"Jarvis, create file summary.txt"`
- `"Jarvis, delete file temporary.txt"` *(Requires spoken confirmation: "Confirm deletion")*

### Web Search
- `"Jarvis, search the web for quantum computing"`
- `"Jarvis, google artificial intelligence breakthroughs"`
- `"Jarvis, look up python tutorials"`

### System Telemetry & Controls
- `"Jarvis, what is the CPU usage"` / `"Check CPU performance"`
- `"Jarvis, what is my RAM usage"` / `"Check memory status"`
- `"Jarvis, how much disk space is left"`
- `"Jarvis, check battery percentage"`
- `"Jarvis, show system status"`
- `"Jarvis, increase volume"` / `"Turn up the sound"`
- `"Jarvis, decrease volume"` / `"Turn down the sound"`
- `"Jarvis, mute audio"` / `"Unmute audio"`
- `"Jarvis, lock the screen"`

### Conversational Deep Learning Model Training
- `"Jarvis, enter training mode"`
- `"When I say launch my browser, open Chrome"` (Adds new training example)
- `"Jarvis, review my training examples"`
- `"Jarvis, delete the last training example"`
- `"Jarvis, train the model"` (Triggers background PyTorch training worker)
- `"Jarvis, evaluate the model"` (Runs evaluation metrics on test split)
- `"Jarvis, tell me the current model version"`
- `"Jarvis, restore the previous model"` (Rolls back to version $N-1$)

---

## 4. Security & Permissions Architecture

### Permission Levels

```
[PRIVILEGED]         (Rank 5) -> OS Shutdown, Reboot, Process Killing
     ↑
[DESTRUCTIVE]        (Rank 4) -> File Deletion, Model Rollback, App Termination (Requires Confirmation)
     ↑
[FILE_MODIFICATION]  (Rank 3) -> Creating Files, Writing Data
     ↑
[STANDARD]           (Rank 2) -> Opening Apps, Web Searches, Volume, Model Training
     ↑
[READ_ONLY]          (Rank 1) -> System Metrics, File Reading, Status Inquiries
```

### Two-Phase Spoken Confirmation
Destructive and privileged operations are never executed immediately.
1. Jarvis registers a `PendingAction` with a unique UUID, timestamp, and a configurable 30-second expiration window.
2. Jarvis responds with a spoken prompt:
   > *"You requested to delete file with parameters {'filename': 'notes.txt'}. This is a sensitive operation. Please say 'confirm' or 'confirm delete_file' to authorize, or 'cancel' to abort. You have 30 seconds."*
3. The user must speak: `"Confirm"`, `"Confirm deletion"`, or `"Cancel"`.
4. If the timer expires or unrelated commands are issued, the pending action is invalidated immediately to prevent replay attacks.

### Path Traversal & Shell Injection Defense
- **Path Traversal Defense**: All file paths are canonicalized (`Path.resolve()`) and checked with `is_safe_path()`. Access outside allowed directories (`Documents`, `Downloads`, `Desktop`, `jarvis/data`) is strictly denied.
- **Shell Injection Defense**: All system subprocesses are executed with `shell=False` using explicit argument lists. Metacharacters (`;`, `&`, `|`, `` ` ``, `$`) are rejected.
- **Loopback Suppression**: While pyttsx3 is synthesizing speech, Jarvis's audio engine mutes microphone input to guarantee the assistant will never interpret its own synthesized voice as a command.

---

## 5. Installation & Setup Instructions

### Prerequisites
- Python 3.11, 3.12, 3.13, or 3.14
- Working microphone and speakers

### Step 1: Clone or Navigate to the Project
```bash
cd "C:\Users\CSE LAB 2\jarvis"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set up the Offline Vosk STT Model (required for voice commands)
Without this model Jarvis can only take typed commands; spoken commands are ignored and a "Voice commands are disabled" warning is shown at startup.
1. Download `vosk-model-small-en-us-0.15` (approx 40 MB) from [Vosk Models](https://alphacephei.com/vosk/models).
2. Extract the model folder to `jarvis/models/vosk-model-small-en-us-0.15` (the folder must directly contain `am/`, `conf/` and `graph/`; a doubled folder from unzipping is detected automatically).
3. The assistant locates it relative to the project folder (not the current directory) and loads it on startup.

Speak the wake word and command together ("Jarvis open Chrome") or say "Jarvis", wait for "At your service", then give the command. Use `python main.py --device <index>` if the wrong microphone is picked.

---

## 6. Running Jarvis

### Graphical Holographic Interface (Default)
Launch the futuristic PyQt6 Arc Reactor desktop dashboard:
```bash
python main.py
```

### Voice Command-Line Mode (CLI)
For headless systems or pure console testing:
```bash
python main.py --cli
```

### Selecting a Specific Microphone Device
To list and select a specific microphone index:
```bash
python main.py --device 1
```

---

## 7. Running the Test Suite

Jarvis includes a comprehensive automated test suite powered by `pytest` covering all modules:
- Voice capture, wake-word detection, device listing, loopback prevention
- PyTorch tokenizer, neural network forward pass, softmax probabilities
- Training pipelines, dataset splitting, checkpointing, versioning, rollback
- Path traversal defense, filename sanitization, command injection rejection
- Confirmation manager lifecycle, token expiration, replay prevention
- Permission hierarchy and unauthorized command rejection
- Application launcher, sandboxed file operations, web search, system telemetry
- End-to-end conversational training workflows

Execute the complete test suite:
```bash
python -m pytest tests -v
```

Expected result:
```text
============================= 28 passed in 5.21s =============================
```

---

## 8. Packaging into a Standalone Desktop Executable

To package Jarvis into a standalone Windows desktop executable using PyInstaller:
```bash
pyinstaller jarvis.spec
```
The compiled application and all dependencies will be located in the `dist/Jarvis/` directory, ready to run without requiring a Python installation.

---

## 9. Troubleshooting & FAQ

1. **Microphone Not Detected / Audio Device Error**:
   - Check Windows Sound Settings and verify microphone permissions are enabled for desktop applications.
   - Run `python -c "import sounddevice as sd; print(sd.query_devices())"` to view device IDs, then launch with `python main.py --device <ID>`.
2. **Text-To-Speech Output Silent**:
   - Verify Windows SAPI5 audio output volume is turned up.
   - Adjust `tts_rate` or `tts_volume` in `config/settings.json`.
3. **Model Retraining Performance**:
   - Jarvis uses PyTorch running on CPU by default (or CUDA if an NVIDIA GPU is available).
   - Training 30 epochs on the baseline dataset typically completes in under 3 seconds in the background without interrupting audio listening.
---

## 10. Optional Offline LLM for Natural-Language Understanding

Jarvis keeps the existing PyTorch intent classifier. When that classifier is uncertain, an optional local Ollama model can help interpret ordinary conversation or propose a registered command. The LLM is not an execution engine: every proposed intent is schema-validated and still passes through Jarvis's command registry, permission checks, and confirmation manager. Sensitive intents (including deletion, shutdown/reboot, app termination, model rollback, training controls, and confirmation/cancellation) are not accepted from the LLM.

1. Install [Ollama](https://ollama.com/) on this computer.
2. Download a compact model locally, for example:
   ```bash
   ollama pull qwen2.5:3b
   ```
3. Confirm the local service responds:
   ```bash
   ollama run qwen2.5:3b
   ```
4. In `config/settings.json`, review `local_llm_enabled`, `local_llm_model`, `local_llm_base_url`, and `local_llm_timeout_seconds`. The configured endpoint must remain on loopback (`127.0.0.1` / `localhost` / `::1`); the client has no cloud fallback.
5. Start Jarvis normally. If Ollama is not available or its output is invalid, Jarvis falls back to the existing PyTorch parser and its clarification behavior.

A 3B quantized model is a practical starting point for a 16 GB RAM machine, but actual memory use and latency depend on quantization, context size, and other applications. The model is downloaded once; inference is local thereafter.

**Security note:** The built-in speaker-verification class is only a hook, not a trained biometric system. If `require_speaker_verification` is enabled without a real verifier, high-trust speaker verification now fails closed instead of pretending to authenticate the speaker. Do not treat voice recognition/transcription as proof of speaker identity.

## 11. System-wide desktop application control

Jarvis now includes an app-agnostic Windows UI Automation layer rather than requiring a custom plugin for every application.

- **Discover and open installed apps:** the launcher searches Start Menu shortcuts, Windows App Paths registry entries, and executable search paths. The old configured aliases still work, but they are no longer the only way to launch an app.
- **Inspect the active window:** ask which application/window is currently focused.
- **Interact by accessible control name:** click a uniquely named UI Automation control in the active window. A confirmation is required before the click.
- **Type dictated text and use safe shortcuts:** Jarvis can type into the focused control and supports a small allowlist of navigation/editing shortcuts. Arbitrary shell execution and coordinate guessing are not part of this interface.
- **Dependency:** on Windows, install the optional UI Automation package with `python -m pip install pywinauto` (it is also declared as a Windows-only requirement).

Example voice requests include “Open Paint”, “What window is active?”, “Click the Save button”, “Type hello world”, and “Press Ctrl+S”. Commands are still parsed through the local intent pipeline and dispatched through the permission/confirmation layers.

**Limitations and safety:** generic automation depends on an application's exposed accessibility tree. Apps with custom-drawn controls, elevated/admin windows, remote desktops, UAC prompts, games, or protected surfaces may need special handling or may not be automatable. Clicking a UI control requires confirmation because the same generic button could trigger a consequential action. This feature needs testing on the target Windows machine; passing CI tests does not prove compatibility with every installed app.

