# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None
jarvis_dir = Path(__file__).resolve().parent

datas = [
    (str(jarvis_dir / 'config' / 'settings.json'), 'config'),
    (str(jarvis_dir / 'data'), 'data'),
    (str(jarvis_dir / 'models'), 'models'),
]

a = Analysis(
    [str(jarvis_dir / 'main.py')],
    pathex=[str(jarvis_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'torch',
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'pyttsx3',
        'pyttsx3.drivers',
        'pyttsx3.drivers.sapi5',
        'sounddevice',
        'vosk',
        'psutil',
        'sqlite3',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='JarvisAssistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Jarvis',
)
