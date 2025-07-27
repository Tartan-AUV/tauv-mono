import sys
if sys.prefix == '/home/gleb/dev/venv-tauv':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/gleb/dev/tauv-mono/install/tauv_common'
