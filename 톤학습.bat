@echo off
rem auto_guitar_tone GUI 런처. 더블클릭하면 cmd 창 없이 GUI가 뜬다.
start "" "%LOCALAPPDATA%\Microsoft\WindowsApps\pythonw.exe" "%~dp0src\gui.py"
