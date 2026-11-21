FROM --platform=$BUILDPLATFORM debian:bookworm-slim AS build
ARG TARGETARCH
RUN case "$TARGETARCH:$(dpkg --print-architecture)" in \
      amd64:amd64) package=g++ ;; \
      amd64:*) package=g++-x86-64-linux-gnu ;; \
      *) exit 1 ;; \
    esac && apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends "$package" && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY main.cpp parquet.hpp build.sh train-profile.sh ./
ARG SHOPPING_PGO=gcc
RUN if [ "$(dpkg --print-architecture)" = amd64 ]; then compiler=g++; \
    else compiler=x86_64-linux-gnu-g++; fi && \
    CXX="$compiler" sh build.sh /shopping

FROM scratch
COPY --from=build /shopping /shopping
USER 1000:1000
ENTRYPOINT ["/shopping"]
