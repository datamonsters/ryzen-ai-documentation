@echo off
setlocal enabledelayedexpansion

rem Define the source and target paths
set "sourcePath=..\examples\models"
set "targetPath=./source/example_cards"
rem Check if the target folder exists; create it if it doesn't
if not exist "%targetPath%" (
    mkdir "%targetPath%"
)
rem Iterate through each subfolder in the source path
for /d %%A in ("%sourcePath%\*") do (
    set "subfolder=%%~nxA"
    rem Check if a README.md file exists in the current subfolder
    if exist "%%A\README.md" (
        rem Copy the content of the README.md file to the new Markdown file
        copy "%%A\README.md" "%targetPath%\!subfolder!.md" > nul
    )
)