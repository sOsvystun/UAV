# Unified Build Script for the UAV Wind-Turbine Inspection Suite
# This script builds all components of the project in the correct order.
# Run this script from the project root directory.

# Exit immediately if a command exits with a non-zero status.
$ErrorActionPreference = "Stop"

Write-Host "Starting unified build for UAV Inspection Suite..."

# 1. Build VISION_Recognition (C++ Component)
Write-Host "--------------------------------------------------"
Write-Host "Building VISION_Recognition..."
Write-Host "--------------------------------------------------"
$visionDir = ".\VISION_Recognition"
$buildDir = "$visionDir\build"

if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir
    Write-Host "Created build directory for VISION_Recognition."
}

Push-Location $buildDir
try {
    cmake .. -G "NMake Makefiles" # Using NMake for broader Windows compatibility, can be changed.
    cmake --build . --config Release
    Write-Host "Successfully built VISION_Recognition."
}
finally {
    Pop-Location
}

# 2. Build UAV_server (Rust Component)
Write-Host "--------------------------------------------------"
Write-Host "Building UAV_server..."
Write-Host "--------------------------------------------------"
$serverDir = ".\UAV_server"
Push-Location $serverDir
try {
    cargo build --release
    Write-Host "Successfully built UAV_server."
}
finally {
    Pop-Location
}


# 3. Build UAV_Controller (.NET MAUI Component)
Write-Host "--------------------------------------------------"
Write-Host "Building UAV_Controller..."
Write-Host "--------------------------------------------------"
$controllerSln = ".\UAV_Controller\UAV_Controller.sln"
try {
    dotnet build $controllerSln -c Release
    Write-Host "Successfully built UAV_Controller."
}
catch {
    Write-Error "Failed to build UAV_Controller. Please ensure the .NET 8 SDK and MAUI workloads are installed."
    exit 1
}


Write-Host "--------------------------------------------------"
Write-Host "All components built successfully!"
Write-Host "--------------------------------------------------"
