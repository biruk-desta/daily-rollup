#!/bin/sh
set -eu

if [ "$(dpkg --print-architecture)" = amd64 ]; then compiler=g++
else compiler=x86_64-linux-gnu-g++; fi

SHOPPING_PGO=off CXX="$compiler" sh build.sh /shopping
mkdir base candidate
cp main.cpp parquet.hpp build.sh train-profile.sh base/
patch -d base -p1 < probe-candidate.patch
cp base/*.cpp base/*.hpp base/*.sh candidate/
patch -d candidate -p1 < probe-round8.patch
chmod +x probe-compiler.sh
export CXX=/build/probe-compiler.sh R8_CXX="$compiler"

build_engine() {
    directory=$1
    arch=$2
    destination=$3
    if SHOPPING_ARCH="$arch" sh "$directory/build.sh" "$destination" > build.log 2>&1; then
        if grep -q 'profile-guided optimization enabled' build.log; then status=1
        else status=0; fi
    elif [ "$arch" != baseline ]; then
        cp /shopping-plain "$destination"
        status=x
    else
        cat build.log >&2
        exit 1
    fi
    cat build.log
}

build_engine base baseline /shopping-base
SHOPPING_PGO=off sh candidate/build.sh /shopping-plain
build_engine candidate baseline /shopping-new
base_status=$status
build_engine candidate v3 /shopping-v3
v3_status=$status
build_engine candidate cascadelake /shopping-cl
printf '%s%s%s\n' "$base_status" "$v3_status" "$status" > /pgo-status
"$compiler" -std=c++20 -O2 -static -s -I candidate probe.cpp -o /probe
