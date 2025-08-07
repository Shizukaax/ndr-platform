#!/bin/bash
# NDR Platform - Cleanup Script for Linux/macOS
# Removes temporary files, caches, and organizes project structure

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Clean Python cache files
clean_python_cache() {
    log_info "🐍 Cleaning Python cache files..."
    
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    
    log_success "Python cache cleaned"
}

# Clean log files
clean_logs() {
    log_info "📋 Cleaning old log files..."
    
    find logs/ -name "*.log.[0-9]*" -delete 2>/dev/null || true
    find . -name "*.tmp" -delete 2>/dev/null || true
    find . -name "*.temp" -delete 2>/dev/null || true
    
    log_success "Log files cleaned"
}

# Clean system files
clean_system_files() {
    log_info "🗂️ Cleaning system files..."
    
    find . -name ".DS_Store" -delete 2>/dev/null || true
    find . -name "Thumbs.db" -delete 2>/dev/null || true
    find . -name "desktop.ini" -delete 2>/dev/null || true
    
    log_success "System files cleaned"
}

# Clean Docker resources
clean_docker() {
    log_info "🐳 Cleaning Docker resources..."
    
    if command -v docker &> /dev/null; then
        # Stop containers
        docker-compose -f guides/deployment/docker-compose.yml down 2>/dev/null || true
        
        # Remove unused images, containers, networks
        docker system prune -f 2>/dev/null || true
        
        # Remove build cache
        rm -f .docker_build_cache 2>/dev/null || true
        
        log_success "Docker resources cleaned"
    else
        log_warning "Docker not found, skipping Docker cleanup"
    fi
}

# Clean application cache
clean_app_cache() {
    log_info "🗃️ Cleaning application cache..."
    
    rm -rf cache/* 2>/dev/null || true
    rm -rf .streamlit 2>/dev/null || true
    
    log_success "Application cache cleaned"
}

# Reset permissions
reset_permissions() {
    log_info "🔐 Resetting permissions..."
    
    # Reset script permissions
    find . -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
    
    # Reset directory permissions
    chmod -R 755 logs models results feedback cache 2>/dev/null || true
    
    log_success "Permissions reset"
}

# Main cleanup function
main() {
    log_info "🧹 Starting NDR Platform Cleanup"
    echo ""
    
    clean_python_cache
    clean_logs
    clean_system_files
    clean_app_cache
    
    # Optional deep clean
    if [ "$1" = "--deep" ]; then
        log_warning "Deep clean mode enabled"
        clean_docker
        
        # Remove virtual environment
        if [ -d "venv" ]; then
            log_warning "Removing virtual environment..."
            rm -rf venv
            log_success "Virtual environment removed"
        fi
        
        # Remove results and models (with confirmation)
        echo ""
        read -p "Remove all results and models? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf results/* models/*.pkl models/*.json
            log_success "Results and models removed"
        fi
    fi
    
    reset_permissions
    
    echo ""
    log_success "✅ Cleanup completed!"
    
    if [ "$1" = "--deep" ]; then
        echo ""
        log_info "After deep clean, you may need to:"
        echo "  1. Run ./scripts/linux/setup.sh to recreate environment"
        echo "  2. Retrain your models"
    fi
}

# Parse arguments
case "${1:-}" in
    "--help"|"-h")
        echo "NDR Platform Cleanup Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo "  --deep        Deep clean (removes venv, docker, with prompts)"
        echo ""
        echo "Examples:"
        echo "  $0            # Standard cleanup"
        echo "  $0 --deep     # Deep cleanup with confirmations"
        ;;
    *)
        main "$1"
        ;;
esac
