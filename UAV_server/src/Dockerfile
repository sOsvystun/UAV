# Stage 1: Builder
FROM rust:1.83 AS builder

WORKDIR /UAV_server

# Install needed packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    git \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/protocolbuffers/protobuf.git /protobuf
ENV PROTOC_INCLUDE=/protobuf/src

# Copy cargo files and pre-build dependencies
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release --locked

# Copy the entire source
COPY . .

# Build the final binary
RUN cargo build --release --locked

# Stage 2: Runtime
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -r -s /bin/bash appuser

WORKDIR /UAV_server

# Copy the binary and .env file from builder stage
COPY --from=builder /UAV_server/target/release/UAV_server .
COPY .env .env

# Switch to non-root user
RUN chown -R appuser:appuser /UAV_server
USER appuser

# Expose the port your app listens on
EXPOSE 5051

CMD ["././UAV_server"]
