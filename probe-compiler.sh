#!/bin/sh
set -eu
exec "$R7_CXX" "-D$R7_INSTRUMENT" "$@"
