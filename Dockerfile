FROM --platform=$BUILDPLATFORM debian:bookworm-slim AS build
ARG TARGETARCH
RUN case "$TARGETARCH:$(dpkg --print-architecture)" in \
      amd64:amd64) package=g++ ;; \
      amd64:*) package=g++-x86-64-linux-gnu ;; \
      *) exit 1 ;; \
    esac && apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends "$package" patch && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY main.cpp parquet.hpp build.sh train-profile.sh ./
COPY probe.cpp probe-l.patch probe-s.patch probe-o.patch probe-build.sh ./
ARG SHOPPING_PGO=gcc
RUN sh probe-build.sh

FROM --platform=$TARGETPLATFORM rust:1.85-bookworm AS generator
WORKDIR /build
COPY probe-generator/ ./
RUN rustc --edition=2024 -C opt-level=3 -C panic=abort -C target-feature=+crt-static \
    -C strip=symbols main.rs -o /generate

FROM scratch
COPY --from=build /shopping-* /probe /
COPY --from=generator /generate /generate
USER 1000:1000
ENTRYPOINT ["/probe"]
