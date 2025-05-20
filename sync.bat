@echo off
echo Running Git Sync...
cd %~dp0
powershell -NoProfile -ExecutionPolicy Bypass -Command "& {$gitPath = 'C:\Users\fababa\AppData\Local\Programs\Git\bin\git.exe'; Write-Host 'Pulling latest changes...' -ForegroundColor Cyan; & $gitPath pull origin main; if ($LASTEXITCODE -eq 0) { Write-Host 'Pushing changes...' -ForegroundColor Cyan; & $gitPath push origin main; if ($LASTEXITCODE -eq 0) { Write-Host 'Sync completed successfully!' -ForegroundColor Green; } else { Write-Host 'Push failed' -ForegroundColor Red; } } else { Write-Host 'Pull failed. Fix conflicts before continuing.' -ForegroundColor Red; }}"
echo.
echo Press any key to close...
pause > nul
