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
COPY probe.cpp probe-base.patch probe-candidate.patch probe-compiler.sh ./
ARG SHOPPING_PGO=gcc
RUN if [ "$(dpkg --print-architecture)" = amd64 ]; then compiler=g++; \
    else compiler=x86_64-linux-gnu-g++; fi && \
    CXX="$compiler" sh build.sh /shopping && \
    mkdir base candidate && \
    cp main.cpp parquet.hpp build.sh train-profile.sh base/ && \
    cp main.cpp parquet.hpp build.sh train-profile.sh candidate/ && \
    patch -d base -p1 < probe-base.patch && patch -d candidate -p1 < probe-candidate.patch && \
    cp candidate/detail_trace.hpp base/ && chmod +x probe-compiler.sh && \
    export CXX=/build/probe-compiler.sh R7_CXX="$compiler" && \
    R7_INSTRUMENT=SHOPPING_PHASE_COUNTERS sh base/build.sh /shopping-base && \
    R7_INSTRUMENT=SHOPPING_PHASE_COUNTERS sh candidate/build.sh /shopping-new && \
    R7_INSTRUMENT=SHOPPING_DETAIL_TRACE sh base/build.sh /shopping-base-detail && \
    R7_INSTRUMENT=SHOPPING_DETAIL_TRACE sh candidate/build.sh /shopping-new-detail && \
    "$compiler" -std=c++20 -O2 -static -s probe.cpp -o /probe

FROM --platform=$TARGETPLATFORM rust:1.85-bookworm AS generator
WORKDIR /build
COPY probe-generator/ ./
RUN rustc --edition=2024 -C opt-level=3 -C panic=abort -C target-feature=+crt-static \
    -C strip=symbols main.rs -o /generate

FROM scratch
COPY --from=build /shopping /shopping-base /shopping-new /shopping-base-detail /shopping-new-detail /probe /
COPY --from=generator /generate /generate
USER 1000:1000
ENTRYPOINT ["/probe"]
