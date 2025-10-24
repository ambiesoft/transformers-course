REM call c:\Linkout\bat\envpython3.bat
REM call  C:\Linkout\bat\envffmpeg.bat
call %~dp0\.env\Scripts\activate
start "" C:\local\VSCode\Code.exe "%~dp0"
