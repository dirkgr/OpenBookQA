#!/usr/bin/env bash

export METRIC=$1
shift
EXPERIMENT=$1
shift

set -e

ls -1 $EXPERIMENT/*.json | \
parallel 'echo {} $(jq .$METRIC {})' | \
sort | \
sed -Ee 's/\.json//' | \
sed -e "s/^[^ ]*\/\+//" | \
csvtool -t " " -u TAB transpose -
