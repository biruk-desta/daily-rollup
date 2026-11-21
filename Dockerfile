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
COPY probe.cpp probe-engine.sed probe-read.patch read_input.hpp ./
ARG SHOPPING_PGO=gcc
RUN if [ "$(dpkg --print-architecture)" = amd64 ]; then compiler=g++; \
    else compiler=x86_64-linux-gnu-g++; fi && export CXX="$compiler" && \
    sh build.sh /shopping && \
    mkdir tuned read-engine && \
    cp parquet.hpp build.sh train-profile.sh tuned/ && \
    cp main.cpp parquet.hpp build.sh train-profile.sh read_input.hpp read-engine/ && \
    sed -f probe-engine.sed main.cpp > tuned/main.cpp && \
    patch -d read-engine -p1 < probe-read.patch && \
    sh tuned/build.sh /shopping-tuned && sh read-engine/build.sh /shopping-read && \
    "$compiler" -std=c++20 -O2 -static -s -pthread probe.cpp -o /probe

FROM --platform=$TARGETPLATFORM rust:1.85-bookworm AS generator
WORKDIR /build
COPY probe-generator/ ./
RUN rustc --edition=2024 -C opt-level=3 -C panic=abort -C target-feature=+crt-static \
    -C strip=symbols main.rs -o /generate

FROM scratch
COPY --from=build /shopping /shopping-tuned /shopping-read /probe /
COPY --from=generator /generate /generate
USER 1000:1000
ENTRYPOINT ["/probe"]
