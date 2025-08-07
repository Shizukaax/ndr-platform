@echo off
REM NDR Platform - Setup Script for Windows
REM Creates directories, sets up environment, and configures the platform

setlocal enabledelayedexpansion

REM Function to log messages
:log_info
echo [INFO] %~1
goto :eof

:log_success
echo [SUCCESS] %~1
goto :eof

:log_warning
echo [WARNING] %~1
goto :eof

:log_error
echo [ERROR] %~1
goto :eof

REM Create directory structure
:create_directories
call :log_info "📁 Creating directory structure..."

set directories=data\examples data\realtime logs models\backups reports results feedback cache config deployment

for %%d in (%directories%) do (
    if not exist "%%d" (
        mkdir "%%d" 2>nul
        call :log_success "Created: %%d"
    )
)
goto :eof

REM Setup Python environment
:setup_python_env
call :log_info "🐍 Setting up Python environment..."

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Python is not installed or not in PATH"
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set python_version=%%v
call :log_info "Python version: !python_version!"

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    call :log_info "Creating virtual environment..."
    python -m venv venv
    if errorlevel 1 (
        call :log_error "Failed to create virtual environment"
        exit /b 1
    )
    call :log_success "Virtual environment created"
)

REM Activate and install dependencies
call :log_info "Installing dependencies..."
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    call :log_error "Failed to install dependencies"
    exit /b 1
)
call :log_success "Dependencies installed"
goto :eof

REM Setup configuration
:setup_config
call :log_info "⚙️ Setting up configuration..."

if not exist "config\config.yaml" (
    if exist "config\config.example.yaml" (
        copy "config\config.example.yaml" "config\config.yaml" >nul
        call :log_success "Created config.yaml from example"
    ) else (
        call :log_warning "No example config found, you'll need to create config.yaml manually"
    )
) else (
    call :log_info "Configuration already exists"
)
goto :eof

REM Check dependencies
:check_dependencies
call :log_info "🔍 Checking system dependencies..."

set dependencies=git curl
set missing=

for %%d in (%dependencies%) do (
    %%d --version >nul 2>&1
    if errorlevel 1 (
        set missing=!missing! %%d
    )
)

if not "!missing!"=="" (
    call :log_warning "Missing dependencies:!missing!"
    call :log_info "Install Git from: https://git-scm.com/download/win"
    call :log_info "Install curl (usually included with Windows 10+)"
) else (
    call :log_success "All dependencies found"
)
goto :eof

REM Main setup function
:main
call :log_info "🚀 Starting NDR Platform Setup"
echo.

call :check_dependencies
call :create_directories
call :setup_config
call :setup_python_env

echo.
call :log_success "✅ NDR Platform setup completed!"
echo.
call :log_info "Next steps:"
echo   1. Review and edit config\config.yaml
echo   2. Add your data files to the data\ directory
echo   3. For Docker deployment: deploy.bat
echo   4. For development: venv\Scripts\activate ^&^& streamlit run run.py
echo   5. Access the platform at: http://localhost:8501
echo.
call :log_info "💡 Tip: Use 'deploy.bat' for production Docker deployment"
call :log_info "💡 Tip: Use 'streamlit run run.py' for development mode"
goto :eof

REM Parse command line arguments
set COMMAND=%1
if "%COMMAND%"=="" goto main
if "%COMMAND%"=="--help" goto show_help
if "%COMMAND%"=="-h" goto show_help
if "%COMMAND%"=="--force" goto force_setup
goto unknown_option

:show_help
echo NDR Platform Setup Script for Windows
echo.
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   --help, -h    Show this help message
echo   --force       Force recreate directories and config
echo.
goto :eof

:force_setup
call :log_warning "Force mode enabled - will recreate existing files"
rmdir /s /q venv 2>nul
del config\config.yaml 2>nul
goto main

:unknown_option
call :log_error "Unknown option: %COMMAND%"
echo Use --help for usage information
exit /b 1
