"""Tests for Individual Execution Plugins.
"""

import tempfile
from pathlib import Path
import pytest

from plugins.applications import ApplicationPlugin
from plugins.desktop_automation import DesktopAutomationPlugin, SAFE_HOTKEYS
from plugins.files import FilePlugin
from plugins.system_info import SystemInfoPlugin
from plugins.web_search import WebSearchPlugin


def test_application_plugin_validation():
    plugin = ApplicationPlugin()

    # Missing parameter
    res = plugin.execute("open_application", {}, {})
    assert res.success is False
    assert "No application name" in res.spoken_response

    # Unapproved application
    res2 = plugin.execute("open_application", {"application": "unapproved_trojan"}, {})
    assert res2.success is False
    assert "approved application list" in res2.spoken_response


def test_file_plugin_sandboxed_operations():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        plugin = FilePlugin()

        class MockSettings:
            def get_allowed_directories(self):
                return [tmp_path]

        context = {"settings": MockSettings()}

        # 1. Create file
        res_create = plugin.execute(
            "create_file",
            {"filename": "hello.txt", "content": "Hello Jarvis World"},
            context,
        )
        assert res_create.success is True
        created_file = tmp_path / "hello.txt"
        assert created_file.exists()
        assert created_file.read_text(encoding="utf-8") == "Hello Jarvis World"

        # 2. Read file
        res_read = plugin.execute("read_file", {"filename": "hello.txt"}, context)
        assert res_read.success is True
        assert "Hello Jarvis World" in res_read.spoken_response

        # 3. List files
        res_list = plugin.execute("list_files", {}, context)
        assert res_list.success is True
        assert "hello.txt" in res_list.data["files"]

        # 4. Delete file
        res_del = plugin.execute("delete_file", {"filename": "hello.txt"}, context)
        assert res_del.success is True
        assert not created_file.exists()


def test_web_search_plugin():
    plugin = WebSearchPlugin()

    # Empty query
    res = plugin.execute("web_search", {"query": ""}, {})
    assert res.success is False

    # Valid query
    res2 = plugin.execute("web_search", {"query": "artificial intelligence breakthroughs"}, {})
    assert res2.success is True
    assert "google.com" in res2.data["url"]
    assert "artificial+intelligence+breakthroughs" in res2.data["url"]


def test_system_info_plugin():
    plugin = SystemInfoPlugin()
    res = plugin.execute("system_info", {"metric": "all"}, {})
    assert res.success is True
    assert "cpu_percent" in res.data
    assert "memory_percent" in res.data
    assert "disk_percent" in res.data
    assert isinstance(res.data["cpu_percent"], float)



def test_application_plugin_can_resolve_apps_outside_configured_aliases(monkeypatch):
    plugin = ApplicationPlugin()
    launched = []
    monkeypatch.setattr(plugin, "_discover_installed_app", lambda name: "/opt/example-editor/example-editor" if name == "example editor" else None)
    monkeypatch.setattr("plugins.applications.subprocess.Popen", lambda args, **kwargs: launched.append((args, kwargs)))
    result = plugin.execute("open_application", {"application": "Example Editor"}, {})
    assert result.success is True
    assert launched
    assert launched[0][0] == ["/opt/example-editor/example-editor"]
    assert launched[0][1]["shell"] is False


def test_desktop_automation_shortcut_allowlist_excludes_arbitrary_keys():
    assert "ctrl+a" in SAFE_HOTKEYS
    assert "alt+tab" in SAFE_HOTKEYS
    assert "ctrl+alt+delete" not in SAFE_HOTKEYS
    assert "win+r" not in SAFE_HOTKEYS


def test_desktop_automation_plugin_registers_app_agnostic_intents():
    plugin = DesktopAutomationPlugin()
    assert {"get_active_window", "click_ui_control", "type_text", "press_hotkey"} <= set(plugin.supported_intents)
    assert plugin.required_permissions["get_active_window"].value == "READ_ONLY"
