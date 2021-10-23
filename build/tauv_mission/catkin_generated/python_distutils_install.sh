#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/tom/workspaces/tauv_ws/src/packages/tauv_mission"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/tom/workspaces/tauv_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/tom/workspaces/tauv_ws/install/lib/python3/dist-packages:/home/tom/workspaces/tauv_ws/build/tauv_mission/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/tom/workspaces/tauv_ws/build/tauv_mission" \
    "/usr/bin/python3" \
    "/home/tom/workspaces/tauv_ws/src/packages/tauv_mission/setup.py" \
    egg_info --egg-base /home/tom/workspaces/tauv_ws/build/tauv_mission \
    build --build-base "/home/tom/workspaces/tauv_ws/build/tauv_mission" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/tom/workspaces/tauv_ws/install" --install-scripts="/home/tom/workspaces/tauv_ws/install/bin"
