# Build stage
# must be >= the channel in rust-toolchain.toml, or the build re-downloads a whole toolchain
FROM rust:1.90 AS builder
WORKDIR /usr/src/r1ver
COPY . .
# The API server is gated behind the `analyze` feature — a default build produces a
# binary that starts up and immediately exits. `demo` and `shortdeck` must match the
# features the published dataset was generated with: they set the abstraction bucket
# counts and the 36-card deck, so a mismatched binary misreads every abstraction.
RUN cargo build --release --locked --features "analyze demo shortdeck"

# Final stage
FROM debian:bookworm-slim
RUN apt-get update && \
    apt-get install -y libssl3 ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/r1ver/target/release/r1ver .
COPY --from=builder /usr/src/r1ver/NOTICE /usr/share/doc/r1ver/NOTICE
# Serve on all interfaces; the platform's PORT overrides the default.
ENV HOST=0.0.0.0
EXPOSE 3002
ENTRYPOINT ["/r1ver"]
