@echo off
REM NDR Platform - Cleanup Script for Windows
REM Removes temporary files, caches, and organizes project structure

setlocal enabledelayedexpansion

:log_info
echo [INFO] %~1
goto :eof

:log_success
echo [SUCCESS] %~1
goto :eof

:log_warning
echo [WARNING] %~1
goto :eof

REM Clean Python cache files
:clean_python_cache
call :log_info "🐍 Cleaning Python cache files..."

for /r %%i in (__pycache__) do (
    if exist "%%i" (
        rmdir /s /q "%%i" 2>nul
    )
)

del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul

for /r %%i in (.pytest_cache) do (
    if exist "%%i" (
        rmdir /s /q "%%i" 2>nul
    )
)

call :log_success "Python cache cleaned"
goto :eof

REM Clean log files
:clean_logs
call :log_info "📋 Cleaning old log files..."

if exist logs (
    del /q logs\*.log.* 2>nul
)
del /s /q *.tmp 2>nul
del /s /q *.temp 2>nul

call :log_success "Log files cleaned"
goto :eof

REM Clean system files
:clean_system_files
call :log_info "🗂️ Cleaning system files..."

del /s /q Thumbs.db 2>nul
del /s /q desktop.ini 2>nul

call :log_success "System files cleaned"
goto :eof

REM Clean Docker resources
:clean_docker
call :log_info "🐳 Cleaning Docker resources..."

docker --version >nul 2>&1
if not errorlevel 1 (
    REM Stop containers
    docker-compose -f deployment\docker-compose.yml down 2>nul
    
    REM Remove unused resources
    docker system prune -f 2>nul
    
    REM Remove build cache
    del .docker_build_cache 2>nul
    
    call :log_success "Docker resources cleaned"
) else (
    call :log_warning "Docker not found, skipping Docker cleanup"
)
goto :eof

REM Clean application cache
:clean_app_cache
call :log_info "🗃️ Cleaning application cache..."

if exist cache (
    del /q cache\* 2>nul
)
if exist .streamlit (
    rmdir /s /q .streamlit 2>nul
)

call :log_success "Application cache cleaned"
goto :eof

REM Main cleanup function
:main
call :log_info "🧹 Starting NDR Platform Cleanup"
echo.

call :clean_python_cache
call :clean_logs
call :clean_system_files
call :clean_app_cache

REM Optional deep clean
if "%1"=="--deep" (
    call :log_warning "Deep clean mode enabled"
    call :clean_docker
    
    REM Remove virtual environment
    if exist venv (
        call :log_warning "Removing virtual environment..."
        rmdir /s /q venv
        call :log_success "Virtual environment removed"
    )
    
    REM Remove results and models (with confirmation)
    echo.
    set /p choice="Remove all results and models? [y/N]: "
    if /i "!choice!"=="y" (
        del /q results\* 2>nul
        del /q models\*.pkl models\*.json 2>nul
        call :log_success "Results and models removed"
    )
)

echo.
call :log_success "✅ Cleanup completed!"

if "%1"=="--deep" (
    echo.
    call :log_info "After deep clean, you may need to:"
    echo   1. Run scripts\windows\setup.bat to recreate environment
    echo   2. Retrain your models
)
goto :eof

REM Parse arguments
set COMMAND=%1
if "%COMMAND%"=="" goto main
if "%COMMAND%"=="--help" goto show_help
if "%COMMAND%"=="-h" goto show_help
if "%COMMAND%"=="--deep" goto main
goto unknown_option

:show_help
echo NDR Platform Cleanup Script for Windows
echo.
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   --help, -h    Show this help message
echo   --deep        Deep clean (removes venv, docker, with prompts)
echo.
echo Examples:
echo   %~nx0            # Standard cleanup
echo   %~nx0 --deep     # Deep cleanup with confirmations
goto :eof

:unknown_option
echo [ERROR] Unknown option: %COMMAND%
echo Use --help for usage information
exit /b 1
