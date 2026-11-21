#!/bin/sh
set -eu

source=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
stage=${2:-/build/engines}
mkdir -p "$stage/dev"
cp "$source/main.cpp" "$source/parquet.hpp" "$source/build.sh" "$source/train-profile.sh" "$stage/dev/"

derive() {
    mkdir -p "$stage/$2"
    cp "$stage/$1"/*.cpp "$stage/$1"/*.hpp "$stage/$1"/*.sh "$stage/$2/"
    patch --batch --fuzz=0 -d "$stage/$2" -p1 < "$source/probe-$2.patch"
}

derive dev l
derive l s

if [ "${1:-}" = --stage ]; then exit 0; fi
if [ "$(dpkg --print-architecture)" = amd64 ]; then CXX=g++
else CXX=x86_64-linux-gnu-g++; fi
export CXX

for engine in dev l s; do
    sh "$stage/$engine/build.sh" "/shopping-$engine"
done
"$CXX" -std=c++20 -O2 -static -s "$source/probe.cpp" -o /probe
