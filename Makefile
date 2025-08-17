# UAV Wind-Turbine Inspection Suite Makefile

.PHONY: help build test clean docker-build docker-up docker-down proto lint format check

# Default target
help:
	@echo "UAV Wind-Turbine Inspection Suite"
	@echo ""
	@echo "Available targets:"
	@echo "  build         - Build all services"
	@echo "  test          - Run all tests"
	@echo "  clean         - Clean build artifacts"
	@echo "  docker-build  - Build Docker images"
	@echo "  docker-up     - Start all services with Docker Compose"
	@echo "  docker-down   - Stop all services"
	@echo "  proto         - Generate protobuf code"
	@echo "  lint          - Run linters"
	@echo "  format        - Format code"
	@echo "  check         - Run all checks (lint, test, build)"

# Build targets
build: build-shared build-services

build-shared:
	@echo "Building shared libraries..."
	cd shared/rust-common && cargo build --release

build-services: build-shared
	@echo "Building services..."
	cd services/gateway && cargo build --release
	cd services/trajectory && cargo build --release
	cd services/detection && cargo build --release
	cd services/criticality && cargo build --release
	cd services/reporting && cargo build --release

build-cpp:
	@echo "Building C++ components..."
	cd VISION_Recognition && mkdir -p build && cd build && cmake .. && make

build-python:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

build-dotnet:
	@echo "Building .NET components..."
	cd UAV_Controller && dotnet build UAV_Controller.sln -c Release

# Test targets
test: test-rust test-cpp test-python test-dotnet

test-rust:
	@echo "Running Rust tests..."
	cd shared/rust-common && cargo test
	cd services/gateway && cargo test
	cd services/trajectory && cargo test
	cd services/detection && cargo test
	cd services/criticality && cargo test
	cd services/reporting && cargo test

test-cpp:
	@echo "Running C++ tests..."
	cd VISION_Recognition/build && make test

test-python:
	@echo "Running Python tests..."
	cd VISION_Fuzzy && python -m pytest tests/

test-dotnet:
	@echo "Running .NET tests..."
	cd UAV_Controller && dotnet test

# Docker targets
docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-up:
	@echo "Starting services..."
	docker-compose up -d

docker-down:
	@echo "Stopping services..."
	docker-compose down

docker-logs:
	docker-compose logs -f

# Development targets
proto:
	@echo "Generating protobuf code..."
	# Rust protobuf generation is handled by build.rs
	cd shared/rust-common && cargo build
	
	# Generate for other languages if needed
	# protoc --python_out=. --grpc_python_out=. shared/proto/*.proto
	# protoc --csharp_out=. --grpc_out=. --plugin=protoc-gen-grpc=grpc_csharp_plugin shared/proto/*.proto

lint: lint-rust lint-python lint-cpp

lint-rust:
	@echo "Linting Rust code..."
	cd shared/rust-common && cargo clippy -- -D warnings
	cd services/gateway && cargo clippy -- -D warnings
	cd services/trajectory && cargo clippy -- -D warnings
	cd services/detection && cargo clippy -- -D warnings
	cd services/criticality && cargo clippy -- -D warnings
	cd services/reporting && cargo clippy -- -D warnings

lint-python:
	@echo "Linting Python code..."
	cd VISION_Fuzzy && flake8 . --max-line-length=100
	cd VISION_Fuzzy && pylint **/*.py

lint-cpp:
	@echo "Linting C++ code..."
	cd VISION_Recognition && find . -name "*.cpp" -o -name "*.h" | xargs clang-format --dry-run --Werror

format: format-rust format-python format-cpp

format-rust:
	@echo "Formatting Rust code..."
	cd shared/rust-common && cargo fmt
	cd services/gateway && cargo fmt
	cd services/trajectory && cargo fmt
	cd services/detection && cargo fmt
	cd services/criticality && cargo fmt
	cd services/reporting && cargo fmt

format-python:
	@echo "Formatting Python code..."
	cd VISION_Fuzzy && black . --line-length=100
	cd VISION_Fuzzy && isort .

format-cpp:
	@echo "Formatting C++ code..."
	cd VISION_Recognition && find . -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# Utility targets
clean:
	@echo "Cleaning build artifacts..."
	cd shared/rust-common && cargo clean
	cd services/gateway && cargo clean
	cd services/trajectory && cargo clean
	cd services/detection && cargo clean
	cd services/criticality && cargo clean
	cd services/reporting && cargo clean
	cd VISION_Recognition && rm -rf build/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	cd UAV_Controller && dotnet clean

check: lint test build
	@echo "All checks passed!"

# Database management
db-setup:
	@echo "Setting up databases..."
	docker-compose up -d postgres
	sleep 5
	docker-compose exec postgres psql -U uav_user -d uav_main -f /docker-entrypoint-initdb.d/init.sql

db-migrate:
	@echo "Running database migrations..."
	cd services/trajectory && sqlx migrate run
	cd services/criticality && sqlx migrate run
	cd services/reporting && sqlx migrate run

# Monitoring setup
monitoring-up:
	@echo "Starting monitoring stack..."
	docker-compose up -d prometheus grafana jaeger

# Development environment
dev-setup: build db-setup monitoring-up
	@echo "Development environment ready!"
	@echo "Services:"
	@echo "  Gateway:    http://localhost:50051"
	@echo "  Grafana:    http://localhost:3000 (admin/admin)"
	@echo "  Prometheus: http://localhost:9091"
	@echo "  Jaeger:     http://localhost:16686"

# Production deployment
deploy-prod:
	@echo "Deploying to production..."
	# Add production deployment commands here
	kubectl apply -f k8s/

# Backup and restore
backup:
	@echo "Creating backup..."
	docker-compose exec postgres pg_dump -U uav_user uav_main > backup_$(shell date +%Y%m%d_%H%M%S).sql

restore:
	@echo "Restoring from backup..."
	@read -p "Enter backup file path: " backup_file; \
	docker-compose exec -T postgres psql -U uav_user -d uav_main < $$backup_file