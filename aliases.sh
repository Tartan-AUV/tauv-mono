#!/bin/sh
dir=$(dirname "$(readlink -f "$0")")

alias tauvmake="pushd $dir &> /dev/null && \
make ros && \
source devel/setup.zsh && \
popd &> /dev/null"

alias tauvsh="pushd $dir &> /dev/null && \
source devel/setup.zsh && \
popd &> /dev/null"

alias tauvclean="pushd $dir &> /dev/null && \
make ros-clean && \
popd &> /dev/null"
