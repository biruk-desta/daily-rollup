#!/bin/sh
set -eu

source_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
compiler=${CXX:-g++}
case "$compiler" in
    */*) compiler=$(CDPATH= cd -- "$(dirname -- "$compiler")" && pwd)/$(basename -- "$compiler") ;;
esac
destination=$1
case "$destination" in
    /*) ;;
    *) destination=$PWD/$destination ;;
esac

if [ "${SHOPPING_NATIVE_BUILD:-0}" = 1 ]; then
    set -- -std=c++20 -O3 -DNDEBUG -pthread
else
    set -- -std=c++20 -O3 -DNDEBUG -pthread -static -flto=1 -s
fi

profile() (
    scratch=$(mktemp -d "${TMPDIR:-/tmp}/shopping-pgo.XXXXXXXX") || exit 1
    trap 'rm -rf -- "$scratch"' EXIT
    cd "$scratch" || exit 1
    sh "$source_dir/train-profile.sh" "$scratch/data" || exit 1

    case "${SHOPPING_PGO:-gcc}" in
        gcc)
            # The dump call in main.cpp is a weak reference: force the archive member in.
            "$compiler" "$@" -fprofile-generate="$scratch/profile" \
                -fprofile-update=atomic -Wl,--undefined=__gcov_dump "$source_dir/main.cpp" -o shopping || exit 1
            ;;
        clang)
            case "$(uname -s)" in Darwin) writer=___llvm_profile_write_file ;; *) writer=__llvm_profile_write_file ;; esac
            "$compiler" "$@" -fprofile-instr-generate -Wl,-u,"$writer" \
                "$source_dir/main.cpp" -o shopping || exit 1
            ;;
        *) exit 1 ;;
    esac

    NCPU=8 LLVM_PROFILE_FILE="$scratch/counts.profraw" \
        "${SHOPPING_PROFILE_RUNNER:-env}" ./shopping "$scratch/data" "$scratch/output" || exit 1
    test -s output/user_daily/part-0.parquet && test -s output/campaign_daily/part-0.parquet || exit 1

    case "${SHOPPING_PGO:-gcc}" in
        gcc)
            test -n "$(find profile -name '*.gcda' -size +0c -print)" || exit 1
            "$compiler" "$@" -fprofile-use="$scratch/profile" -fprofile-partial-training \
                -Werror=missing-profile -Werror=coverage-mismatch "$source_dir/main.cpp" -o shopping || exit 1
            ;;
        clang)
            "${LLVM_PROFDATA:-llvm-profdata}" merge -o counts.profdata counts.profraw || exit 1
            "$compiler" "$@" -fprofile-instr-use=counts.profdata \
                "$source_dir/main.cpp" -o shopping || exit 1
            ;;
    esac
    mv -- shopping "$destination"
)

if [ "${SHOPPING_PGO:-gcc}" != off ] && profile "$@"; then
    echo "shopping build: profile-guided optimization enabled"
else
    echo "shopping build: plain optimization" >&2
    "$compiler" "$@" "$source_dir/main.cpp" -o "$destination"
fi
