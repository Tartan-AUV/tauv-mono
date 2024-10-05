#!/bin/bash

# Check if the directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <label studio root directory>"
    exit 1
fi

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$1"

label-studio
