#!/bin/bash

FIRMWARE_DIR="$(pwd)/../../../"

cp -r "$FIRMWARE_DIR/flatcc/include/flatcc" "$FIRMWARE_DIR/rtvc/Middlewares/flatcc/include/"

cd ./include

"$FIRMWARE_DIR"/flatcc/bin/flatcc -rwc "$FIRMWARE_DIR"/schemas/rtvc.fbs
