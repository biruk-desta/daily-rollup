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
COPY main.cpp parquet.hpp detail_trace.hpp csv_index.hpp large_memory.hpp scatter.hpp group_sort.hpp session_buckets.hpp isa.hpp dispatch.cpp build.sh train-profile.sh ./
ARG SHOPPING_PGO=gcc
ARG SHOPPING_ARCH=baseline
RUN if [ "$(dpkg --print-architecture)" = amd64 ]; then compiler=g++; \
    else compiler=x86_64-linux-gnu-g++; fi && \
    mkdir /binout && export CXX="$compiler" && \
    if [ "$SHOPPING_ARCH" = baseline ]; then sh build.sh /binout/shopping; \
    else SHOPPING_ARCH=baseline SHOPPING_PGO=off sh build.sh /binout/shopping-plain && \
         (sh build.sh /binout/shopping-tuned || cp /binout/shopping-plain /binout/shopping-tuned) && \
         "$compiler" -std=c++20 -O2 -static -s -DSHOPPING_VARIANT=\""$SHOPPING_ARCH"\" dispatch.cpp -o /binout/shopping; \
    fi

FROM scratch
COPY --from=build /binout/ /
USER 1000:1000
ENTRYPOINT ["/shopping"]
