from security.confirmation_manager import ConfirmationManager
from security.authentication import VoiceAuthenticator

def manager():
    calls = []
    cm = ConfirmationManager(default_timeout_seconds=10)
    cm.request_confirmation(
        intent="delete_file",
        target_description="delete report.txt",
        callback=lambda: calls.append("executed") or True,
        parameters={"filename": "report.txt"},
    )
    return cm, calls

def test_target_mention_does_not_authorize():
    cm, calls = manager()
    ok, _, _ = cm.verify_and_execute("delete file report.txt")
    assert not ok
    assert calls == []
    assert cm.has_pending()

def test_substring_no_does_not_cancel():
    cm, calls = manager()
    ok, _, _ = cm.verify_and_execute("notebook")
    assert not ok
    assert calls == []
    assert cm.has_pending()

def test_exact_confirm_executes_once():
    cm, calls = manager()
    ok, _, _ = cm.verify_and_execute("confirm")
    assert ok
    assert calls == ["executed"]
    ok2, _, _ = cm.verify_and_execute("confirm")
    assert not ok2
    assert calls == ["executed"]

def test_exact_cancel_does_not_execute():
    cm, calls = manager()
    ok, _, _ = cm.verify_and_execute("cancel")
    assert not ok
    assert calls == []
    assert not cm.has_pending()

def test_required_speaker_verification_fails_closed_without_real_model():
    auth = VoiceAuthenticator(require_speaker_verification=True)
    ok, confidence, _ = auth.verify_speaker(b"some audio")
    assert not ok
    assert confidence == 0.0
