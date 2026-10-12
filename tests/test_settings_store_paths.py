import json
from pathlib import Path

from storage.settings_store import SettingsStore


def test_allowed_directories_expand_home_and_project_relative_paths(tmp_path):
    project = tmp_path / "project"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    settings_path = config_dir / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "allowed_file_directories": [
                    "data",
                    "~/Documents",
                    str(data_dir),
                ]
            }
        ),
        encoding="utf-8",
    )

    settings = SettingsStore(settings_path)
    directories = settings.get_allowed_directories()

    assert directories[0] == data_dir.resolve()
    assert Path.home() / "Documents" in directories
    assert directories.count(data_dir.resolve()) == 1
