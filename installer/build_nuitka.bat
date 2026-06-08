@echo off
REM Build Sonex Setup Wizard as a standalone Windows .exe (no console window)
REM Requires: pip install nuitka ordered-set zstandard

set SCRIPT=%~dp0setup_gui.py
set OUT=%~dp0dist

python -m nuitka ^
  --standalone ^
  --windows-disable-console ^
  --enable-plugin=tk-inter ^
  --output-dir="%OUT%" ^
  --output-filename=SonexSetup ^
  --company-name="Sonex" ^
  --product-name="Sonex Setup" ^
  --file-version=0.4.0.0 ^
  --product-version=0.4.0.0 ^
  --include-module=installer.core ^
  "%SCRIPT%"

echo.
echo Built: %OUT%\SonexSetup.exe
pause
