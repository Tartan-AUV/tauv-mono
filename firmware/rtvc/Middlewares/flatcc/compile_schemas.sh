#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$SCRIPT_DIR/../../../../"
FLATCC_INCLUDE="$REPO_DIR/firmware/rtvc/Middlewares/flatcc/include/flatcc"

mkdir -p "$FLATCC_INCLUDE"
cp -r "$REPO_DIR/firmware/flatcc/include/flatcc" "$REPO_DIR/firmware/rtvc/Middlewares/flatcc/include"

cd ./include

"$REPO_DIR"/firmware/flatcc/bin/flatcc -a "$REPO_DIR"/packages/tauv_vehicle/schemas/eth_msg_rtvc_jetson.fbs
"$REPO_DIR"/firmware/flatcc/bin/flatcc -a "$REPO_DIR"/packages/tauv_vehicle/schemas/eth_msg_jetson_rtvc.fbs
