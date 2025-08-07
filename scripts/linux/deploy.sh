#!/bin/bash
# NDR Platform - Smart Deployment Script
# Automatically handles incremental builds and full rebuilds as needed

set -e  # Exit on error

# Configuration
IMAGE_NAME="ndr-platform"
CONTAINER_NAME="ndr-platform"
COMPOSE_FILE="guides/deployment/docker-compose.yml"
DOCKERFILE="guides/deployment/Dockerfile"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
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

# Function to check if rebuild is needed
needs_rebuild() {
    # Check if container exists
    if ! docker container inspect $CONTAINER_NAME >/dev/null 2>&1; then
        log_info "Container doesn't exist - full build needed"
        return 0
    fi
    
    # Check if image exists
    if ! docker image inspect $IMAGE_NAME >/dev/null 2>&1; then
        log_info "Image doesn't exist - full build needed"
        return 0
    fi
    
    # Check if requirements.txt changed
    if [ "requirements.txt" -nt ".docker_build_cache" ]; then
        log_info "requirements.txt modified - full rebuild needed"
        return 0
    fi
    
    # Check if Dockerfile changed
    if [ "$DOCKERFILE" -nt ".docker_build_cache" ]; then
        log_info "Dockerfile modified - full rebuild needed"
        return 0
    fi
    
    # Check if core dependencies changed
    if find core/ -name "*.py" -newer ".docker_build_cache" | grep -q .; then
        log_info "Core Python files modified - checking if restart is sufficient"
        # For core changes, usually just restart is enough due to volume mounts
        return 1
    fi
    
    # Check if app files changed
    if find app/ -name "*.py" -newer ".docker_build_cache" | grep -q .; then
        log_info "App files modified - restart should be sufficient"
        return 1
    fi
    
    log_info "No significant changes detected"
    return 1
}

# Function to perform smart deployment
smart_deploy() {
    log_info "🚀 Starting NDR Platform Smart Deployment"
    
    # Create build cache file if it doesn't exist
    if [ ! -f ".docker_build_cache" ]; then
        touch ".docker_build_cache"
    fi
    
    if needs_rebuild; then
        log_warning "📦 Full rebuild required"
        
        # Stop existing containers
        log_info "Stopping existing containers..."
        docker-compose -f $COMPOSE_FILE down 2>/dev/null || true
        
        # Remove old image to ensure clean build
        docker rmi $IMAGE_NAME 2>/dev/null || true
        
        # Build with no cache to ensure fresh build
        log_info "Building new image (no cache)..."
        docker-compose -f $COMPOSE_FILE build --no-cache
        
        # Update build cache
        touch ".docker_build_cache"
        
    else
        log_info "♻️  Incremental deployment - restarting containers"
        
        # Just restart containers (code changes are reflected due to volume mounts in dev)
        docker-compose -f $COMPOSE_FILE restart
    fi
    
    # Start services
    log_info "Starting services..."
    docker-compose -f $COMPOSE_FILE up -d
    
    # Wait for health check
    log_info "Waiting for health check..."
    sleep 10
    
    # Check if service is healthy
    if docker-compose -f $COMPOSE_FILE ps | grep -q "healthy"; then
        log_success "✅ NDR Platform deployed successfully!"
        log_info "🌐 Application available at: http://localhost:8501"
    else
        log_error "❌ Deployment may have issues. Check logs:"
        docker-compose -f $COMPOSE_FILE logs --tail=20 ndr-platform
    fi
}

# Function to show deployment status
show_status() {
    log_info "📊 NDR Platform Status"
    echo ""
    
    if docker-compose -f $COMPOSE_FILE ps | grep -q "Up"; then
        log_success "✅ Service is running"
        docker-compose -f $COMPOSE_FILE ps
        echo ""
        log_info "🌐 Application: http://localhost:8501"
        log_info "📊 Health check: http://localhost:8501/_stcore/health"
    else
        log_warning "⚠️  Service is not running"
        docker-compose -f $COMPOSE_FILE ps
    fi
}

# Function to clean up
cleanup() {
    log_info "🧹 Cleaning up Docker resources"
    
    # Stop and remove containers
    docker-compose -f $COMPOSE_FILE down
    
    # Remove image
    docker rmi $IMAGE_NAME 2>/dev/null || true
    
    # Remove build cache
    rm -f ".docker_build_cache"
    
    # Prune unused resources
    docker system prune -f
    
    log_success "✅ Cleanup completed"
}

# Function to show logs
show_logs() {
    log_info "📋 Showing NDR Platform logs"
    docker-compose -f $COMPOSE_FILE logs -f ndr-platform
}

# Function to show help
show_help() {
    echo "NDR Platform Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy    Smart deployment (default) - only rebuilds when necessary"
    echo "  rebuild   Force full rebuild and deploy"
    echo "  restart   Restart containers without rebuild"
    echo "  status    Show deployment status"
    echo "  logs      Show application logs"
    echo "  stop      Stop all services"
    echo "  cleanup   Stop services and clean up resources"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deploy     # Smart deployment"
    echo "  $0 rebuild    # Force full rebuild"
    echo "  $0 status     # Check status"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        smart_deploy
        ;;
    "rebuild")
        log_info "🔨 Forcing full rebuild"
        rm -f ".docker_build_cache"
        smart_deploy
        ;;
    "restart")
        log_info "♻️  Restarting containers"
        docker-compose -f $COMPOSE_FILE restart
        show_status
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs
        ;;
    "stop")
        log_info "⏹️  Stopping services"
        docker-compose -f $COMPOSE_FILE down
        log_success "✅ Services stopped"
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
