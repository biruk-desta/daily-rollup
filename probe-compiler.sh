#!/bin/sh
set -eu
exec "$R8_CXX" -DSHOPPING_PHASE_COUNTERS "$@"
