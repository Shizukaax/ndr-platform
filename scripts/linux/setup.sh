#!/bin/bash
# NDR Platform - Setup Script for Linux/macOS
# Creates directories, sets up environment, and configures the platform

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

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create directory structure
create_directories() {
    log_info "📁 Creating directory structure..."
    
    directories=(
        "data/examples"
        "data/realtime"
        "logs"
        "models/backups"
        "reports"
        "results"
        "feedback"
        "cache"
        "config"
        "deployment"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log_success "Created: $dir"
    done
}

# Setup Python environment
setup_python_env() {
    log_info "🐍 Setting up Python environment..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    log_info "Python version: $python_version"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
        log_success "Virtual environment created"
    fi
    
    # Activate and install dependencies
    log_info "Installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    log_success "Dependencies installed"
}

# Setup configuration
setup_config() {
    log_info "⚙️ Setting up configuration..."
    
    if [ ! -f "config/config.yaml" ]; then
        if [ -f "config/config.example.yaml" ]; then
            cp config/config.example.yaml config/config.yaml
            log_success "Created config.yaml from example"
        else
            log_warning "No example config found, you'll need to create config.yaml manually"
        fi
    else
        log_info "Configuration already exists"
    fi
}

# Set permissions
set_permissions() {
    log_info "🔐 Setting permissions..."
    
    # Make scripts executable
    find . -name "*.sh" -exec chmod +x {} \;
    chmod +x scripts/linux/*.sh 2>/dev/null || true
    
    # Set directory permissions
    chmod -R 755 logs models results feedback cache 2>/dev/null || true
    
    log_success "Permissions set"
}

# Check dependencies
check_dependencies() {
    log_info "🔍 Checking system dependencies..."
    
    dependencies=("git" "curl" "wget")
    missing=()
    
    for dep in "${dependencies[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_warning "Missing dependencies: ${missing[*]}"
        log_info "Install with: sudo apt-get install ${missing[*]} (Ubuntu/Debian)"
        log_info "Install with: brew install ${missing[*]} (macOS)"
    else
        log_success "All dependencies found"
    fi
}

# Main setup function
main() {
    log_info "🚀 Starting NDR Platform Setup"
    echo ""
    
    check_dependencies
    create_directories
    setup_config
    setup_python_env
    set_permissions
    
    echo ""
    log_success "✅ NDR Platform setup completed!"
    echo ""
    log_info "Next steps:"
    echo "  1. Review and edit config/config.yaml"
    echo "  2. Add your data files to the data/ directory"
    echo "  3. For Docker deployment: ./deploy.sh"
    echo "  4. For development: source venv/bin/activate && streamlit run run.py"
    echo "  5. Access the platform at: http://localhost:8501"
    echo ""
    log_info "💡 Tip: Use './deploy.sh' for production Docker deployment"
    log_info "💡 Tip: Use 'streamlit run run.py' for development mode"
}

# Parse command line arguments
case "${1:-}" in
    "--help"|"-h")
        echo "NDR Platform Setup Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo "  --force       Force recreate directories and config"
        echo ""
        ;;
    "--force")
        log_warning "Force mode enabled - will recreate existing files"
        rm -rf venv config/config.yaml 2>/dev/null || true
        main
        ;;
    "")
        main
        ;;
    *)
        log_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
