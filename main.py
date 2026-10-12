"""Main Entry Point for JARVIS — Voice-Controlled Deep Learning Desktop Assistant.
Initializes storage, PyTorch neural models, security pipelines, audio manager, and the PyQt6 interface.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from PyQt6.QtWidgets import QApplication

from ai.model_registry import ModelRegistry
from ai.predictor import IntentPredictor
from ai.trainer import ModelTrainer
from ai.local_llm import LocalLLMRouter, OllamaLocalRuntime
from config import DATABASE_PATH, LOGS_DIR, SETTINGS_PATH
from execution.command_registry import CommandRegistry
from execution.automation_engine import AutomationEngine
from execution.dispatcher import CommandDispatcher
from execution.plugin_manager import PluginManager
from nlp.intent_parser import IntentParser
from security.audit_logger import AuditLogger
from security.confirmation_manager import ConfirmationManager
from security.permissions import PermissionLevel
from storage.database import DatabaseManager
from storage.assistant_memory import AssistantMemory
from storage.settings_store import SettingsStore
from storage.training_store import TrainingStore
from ui.main_window import JarvisMainWindow
from voice.audio_capture import AudioCaptureEngine
from voice.audio_manager import AudioManager
from voice.speech_output import SpeechOutputEngine
from voice.speech_recognizer import OfflineSpeechRecognizer
from voice.wake_word import WakeWordDetector


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "jarvis.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def initialize_system():
    """Initializes and connects all architectural layers."""
    setup_logging()
    logger = logging.getLogger("JarvisInit")
    logger.info("Initializing Jarvis Voice-Controlled Desktop Assistant...")

    # 1. Database & Settings
    db = DatabaseManager(DATABASE_PATH)
    settings = SettingsStore(SETTINGS_PATH)
    assistant_memory = AssistantMemory(db) if settings.get("assistant_memory_enabled", True) else None

    # 2. Dataset & Baseline Seeding
    training_store = TrainingStore(db)
    inserted = training_store.seed_default_examples()
    if inserted > 0:
        logger.info(f"Seeded {inserted} baseline training examples.")

    # 3. Model Registry & Initial Training if needed
    model_registry = ModelRegistry(db_manager=db)
    if model_registry.get_latest_version() == 0:
        logger.info("No prior trained model found. Training initial baseline model (v1)...")
        examples = training_store.get_all_examples(approved_only=True)
        trainer = ModelTrainer(registry=model_registry)
        res = trainer.train(examples, epochs=30)
        logger.info(f"Initial model training result: {res.summary_message}")

    # 4. Neural Intent Predictor & NLP Parser
    confidence_thresh = settings.get("nlp_confidence_threshold", 0.65)
    predictor = IntentPredictor(registry=model_registry, confidence_threshold=confidence_thresh)

    # 4b. Optional fully-local LLM fallback; no cloud endpoint or fallback is configured.
    cmd_registry = CommandRegistry()
    llm_router = None
    if settings.get("local_llm_enabled", False):
        try:
            runtime = OllamaLocalRuntime(
                model=settings.get("local_llm_model", "qwen2.5:3b"),
                base_url=settings.get("local_llm_base_url", "http://127.0.0.1:11434"),
                timeout_seconds=float(settings.get("local_llm_timeout_seconds", 8)),
            )
            llm_router = LocalLLMRouter(runtime, cmd_registry.all_intents().keys())
            logger.info("Local LLM fallback enabled with model %s.", runtime.model)
        except Exception:
            logger.exception("Could not initialize local LLM; continuing with PyTorch-only parsing.")
    intent_parser = IntentParser(
        predictor=predictor,
        confidence_threshold=confidence_thresh,
        llm_router=llm_router,
        command_registry=cmd_registry,
        assistant_memory=assistant_memory,
    )

    # 5. Security & Confirmation
    confirm_timeout = settings.get("confirmation_timeout_seconds", 30.0)
    confirmation_mgr = ConfirmationManager(default_timeout_seconds=confirm_timeout)
    audit = AuditLogger(db_manager=db)

    # 6. Execution & Plugins
    enabled_plugins = settings.get("plugins_enabled", None)
    plugin_mgr = PluginManager(enabled_plugin_ids=enabled_plugins)

    context = {
        "settings": settings,
        "assistant_memory": assistant_memory,
        "training_store": training_store,
        "assistant_memory": assistant_memory,
        "automation_engine": automation_engine,
        "model_registry": model_registry,
        "predictor": predictor,
        "db_manager": db,
    }

    automation_engine = AutomationEngine(assistant_memory, cmd_registry) if assistant_memory is not None else None
    context["automation_engine"] = automation_engine

    dispatcher = CommandDispatcher(
        command_registry=cmd_registry,
        plugin_manager=plugin_mgr,
        confirmation_manager=confirmation_mgr,
        audit_logger=audit,
        current_permission_level=PermissionLevel.PRIVILEGED,
        context=context,
    )

    # 7. Voice Pipeline
    sample_rate = settings.get("sample_rate", 16000)
    device_idx = settings.get("audio_device_index", None)
    audio_capture = AudioCaptureEngine(sample_rate=sample_rate, device_index=device_idx)

    wake_word = settings.get("wake_word", "jarvis")
    wake_aliases = settings.get("wake_word_aliases", ["jarvis", "hey jarvis", "okay jarvis"])
    wake_detector = WakeWordDetector(wake_word=wake_word, aliases=wake_aliases)

    vosk_path = settings.get("vosk_model_path", "models/vosk-model-small-en-us-0.15")
    speech_recognizer = OfflineSpeechRecognizer(model_path=vosk_path, sample_rate=sample_rate)

    tts_rate = settings.get("tts_rate", 180)
    tts_vol = settings.get("tts_volume", 1.0)
    speech_output = SpeechOutputEngine(rate=tts_rate, volume=tts_vol)

    audio_manager = AudioManager(
        audio_capture=audio_capture,
        wake_detector=wake_detector,
        speech_recognizer=speech_recognizer,
        speech_output=speech_output,
    )

    return {
        "settings": settings,
        "db": db,
        "training_store": training_store,
        "model_registry": model_registry,
        "predictor": predictor,
        "intent_parser": intent_parser,
        "dispatcher": dispatcher,
        "audio_manager": audio_manager,
    }


def main():
    parser = argparse.ArgumentParser(description="JARVIS Desktop Assistant")
    parser.add_argument("--cli", action="store_true", help="Run in CLI voice mode without PyQt GUI")
    parser.add_argument("--device", type=int, default=None, help="Microphone device index")
    args = parser.parse_args()

    system = initialize_system()
    audio_mgr = system["audio_manager"]
    dispatcher = system["dispatcher"]
    intent_parser = system["intent_parser"]
    training_store = system["training_store"]
    settings = system["settings"]

    if args.device is not None:
        audio_mgr.select_device(args.device)

    # Start audio capture
    mic_ok = audio_mgr.start()
    if not mic_ok:
        print("\n[WARNING] Could not open the microphone. Check the device with --device <index>.\n")
    if not audio_mgr.voice_ready:
        print(f"\n[WARNING] Voice commands are disabled: {audio_mgr.voice_status}\n")
        audio_mgr.speak("Jarvis online, but speech recognition is unavailable. Check the log for details.")
    else:
        audio_mgr.speak("Jarvis online and systems fully functional.")

    if args.cli:
        print("\n=== JARVIS CLI VOICE MODE ===")
        print("Say 'Jarvis' followed by your command, or type commands below. Type 'exit' to quit.\n")

        def on_cli_command(cmd_text):
            print(f"\n[USER]: {cmd_text}")
            cmd_obj = intent_parser.parse(cmd_text)
            resp = dispatcher.dispatch(cmd_obj)
            print(f"[JARVIS]: {resp.spoken_feedback}")
            audio_mgr.speak(resp.spoken_feedback)

        audio_mgr.on_command_ready = on_cli_command

        try:
            while True:
                user_input = input("jarvis> ").strip()
                if user_input.lower() in {"exit", "quit"}:
                    break
                if user_input:
                    print("Text command dispatch is disabled. Speak 'Jarvis' followed by your command.")
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            audio_mgr.stop()
            print("Jarvis shut down safely.")
    else:
        # Launch PyQt6 GUI
        app = QApplication(sys.argv)
        app.setApplicationName("JARVIS Desktop Assistant")

        window = JarvisMainWindow(
            audio_manager=audio_mgr,
            dispatcher=dispatcher,
            intent_parser=intent_parser,
            training_store=training_store,
            settings_store=settings,
        )
        window.show()
        if not audio_mgr.voice_ready:
            window.txt_transcript.appendPlainText(f"[SYSTEM] Voice commands are disabled: {audio_mgr.voice_status}\n")

        try:
            exit_code = app.exec()
        finally:
            audio_mgr.stop()
            sys.exit(exit_code)


if __name__ == "__main__":
    main()
