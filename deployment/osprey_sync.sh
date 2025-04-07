#!/bin/bash
rsync -av --exclude-from=./.syncignore --no-i-r --info=progress2 ./ tauv@10.0.0.20:~/tauv-mono
