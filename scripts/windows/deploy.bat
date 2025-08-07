@echo off
REM NDR Platform - Smart Deployment Script for Windows
REM Automatically handles incremental builds and full rebuilds as needed

setlocal enabledelayedexpansion

REM Configuration
set IMAGE_NAME=ndr-platform
set CONTAINER_NAME=ndr-platform
set COMPOSE_FILE=deployment\docker-compose.yml
set DOCKERFILE=deployment\Dockerfile

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

REM Function to check if rebuild is needed
:needs_rebuild
call :log_info "Checking if rebuild is needed..."

REM Check if container exists
docker container inspect %CONTAINER_NAME% >nul 2>&1
if errorlevel 1 (
    call :log_info "Container doesn't exist - full build needed"
    set REBUILD_NEEDED=1
    goto :eof
)

REM Check if image exists
docker image inspect %IMAGE_NAME% >nul 2>&1
if errorlevel 1 (
    call :log_info "Image doesn't exist - full build needed"
    set REBUILD_NEEDED=1
    goto :eof
)

REM Check if requirements.txt is newer than cache
if exist .docker_build_cache (
    for /f %%i in ('powershell -command "(Get-Item requirements.txt).LastWriteTime -gt (Get-Item .docker_build_cache).LastWriteTime"') do set NEWER=%%i
    if "!NEWER!"=="True" (
        call :log_info "requirements.txt modified - full rebuild needed"
        set REBUILD_NEEDED=1
        goto :eof
    )
)

REM Check if Dockerfile is newer than cache
if exist .docker_build_cache (
    for /f %%i in ('powershell -command "(Get-Item %DOCKERFILE%).LastWriteTime -gt (Get-Item .docker_build_cache).LastWriteTime"') do set NEWER=%%i
    if "!NEWER!"=="True" (
        call :log_info "Dockerfile modified - full rebuild needed"
        set REBUILD_NEEDED=1
        goto :eof
    )
)

call :log_info "No significant changes detected - restart should be sufficient"
set REBUILD_NEEDED=0
goto :eof

REM Function to perform smart deployment
:smart_deploy
call :log_info "🚀 Starting NDR Platform Smart Deployment"

REM Create build cache file if it doesn't exist
if not exist .docker_build_cache (
    echo. > .docker_build_cache
)

call :needs_rebuild

if !REBUILD_NEEDED!==1 (
    call :log_warning "📦 Full rebuild required"
    
    REM Stop existing containers
    call :log_info "Stopping existing containers..."
    docker-compose -f %COMPOSE_FILE% down 2>nul
    
    REM Remove old image to ensure clean build
    docker rmi %IMAGE_NAME% 2>nul
    
    REM Build with no cache to ensure fresh build
    call :log_info "Building new image (no cache)..."
    docker-compose -f %COMPOSE_FILE% build --no-cache
    if errorlevel 1 (
        call :log_error "Build failed!"
        exit /b 1
    )
    
    REM Update build cache
    echo. > .docker_build_cache
    
) else (
    call :log_info "♻️ Incremental deployment - restarting containers"
    
    REM Just restart containers
    docker-compose -f %COMPOSE_FILE% restart
)

REM Start services
call :log_info "Starting services..."
docker-compose -f %COMPOSE_FILE% up -d
if errorlevel 1 (
    call :log_error "Failed to start services!"
    exit /b 1
)

REM Wait for health check
call :log_info "Waiting for health check..."
timeout /t 10 >nul

REM Check if service is healthy
docker-compose -f %COMPOSE_FILE% ps | findstr "healthy" >nul
if not errorlevel 1 (
    call :log_success "✅ NDR Platform deployed successfully!"
    call :log_info "🌐 Application available at: http://localhost:8501"
) else (
    call :log_error "❌ Deployment may have issues. Check logs:"
    docker-compose -f %COMPOSE_FILE% logs --tail=20 ndr-platform
)
goto :eof

REM Function to show deployment status
:show_status
call :log_info "📊 NDR Platform Status"
echo.

docker-compose -f %COMPOSE_FILE% ps | findstr "Up" >nul
if not errorlevel 1 (
    call :log_success "✅ Service is running"
    docker-compose -f %COMPOSE_FILE% ps
    echo.
    call :log_info "🌐 Application: http://localhost:8501"
    call :log_info "📊 Health check: http://localhost:8501/_stcore/health"
) else (
    call :log_warning "⚠️ Service is not running"
    docker-compose -f %COMPOSE_FILE% ps
)
goto :eof

REM Function to clean up
:cleanup
call :log_info "🧹 Cleaning up Docker resources"

REM Stop and remove containers
docker-compose -f %COMPOSE_FILE% down

REM Remove image
docker rmi %IMAGE_NAME% 2>nul

REM Remove build cache
if exist .docker_build_cache del .docker_build_cache

REM Prune unused resources
docker system prune -f

call :log_success "✅ Cleanup completed"
goto :eof

REM Function to show logs
:show_logs
call :log_info "📋 Showing NDR Platform logs"
docker-compose -f %COMPOSE_FILE% logs -f ndr-platform
goto :eof

REM Function to show help
:show_help
echo NDR Platform Deployment Script for Windows
echo.
echo Usage: %~nx0 [COMMAND]
echo.
echo Commands:
echo   deploy    Smart deployment (default) - only rebuilds when necessary
echo   rebuild   Force full rebuild and deploy
echo   restart   Restart containers without rebuild
echo   status    Show deployment status
echo   logs      Show application logs
echo   stop      Stop all services
echo   cleanup   Stop services and clean up resources
echo   help      Show this help message
echo.
echo Examples:
echo   %~nx0 deploy     # Smart deployment
echo   %~nx0 rebuild    # Force full rebuild
echo   %~nx0 status     # Check status
goto :eof

REM Main script logic
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=deploy

if "%COMMAND%"=="deploy" (
    call :smart_deploy
) else if "%COMMAND%"=="rebuild" (
    call :log_info "🔨 Forcing full rebuild"
    if exist .docker_build_cache del .docker_build_cache
    call :smart_deploy
) else if "%COMMAND%"=="restart" (
    call :log_info "♻️ Restarting containers"
    docker-compose -f %COMPOSE_FILE% restart
    call :show_status
) else if "%COMMAND%"=="status" (
    call :show_status
) else if "%COMMAND%"=="logs" (
    call :show_logs
) else if "%COMMAND%"=="stop" (
    call :log_info "⏹️ Stopping services"
    docker-compose -f %COMPOSE_FILE% down
    call :log_success "✅ Services stopped"
) else if "%COMMAND%"=="cleanup" (
    call :cleanup
) else if "%COMMAND%"=="help" (
    call :show_help
) else if "%COMMAND%"=="-h" (
    call :show_help
) else if "%COMMAND%"=="--help" (
    call :show_help
) else (
    call :log_error "Unknown command: %COMMAND%"
    call :show_help
    exit /b 1
)
