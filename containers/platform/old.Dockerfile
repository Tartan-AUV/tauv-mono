FROM dustynv/ros:noetic-pytorch-l4t-r35.1.0

WORKDIR /workspace

# This looks funky
RUN python3 -m pip install -U pip && \ 
    python3 -m pip install --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/ depthai

RUN sudo apt update && apt install -y \
    ros-noetic-gazebo-ros-control \
    ros-noetic-fkie-multimaster \
    ros-noetic-imu-transformer \
    ros-noetic-jsk-recognition-msgs \
    ros-noetic-vision-msgs \
    ros-noetic-phidgets-ik \
    ros-noetic-imu-filter-madgwick \
    libi2c-dev \
    ros-noetic-image-transport \
    ros-noetic-robot-localization \
    ros-noetic-catkin \
    ros-noetic-xacro && \
    cmake && \
    apt-get clean

RUN sudo apt update && apt install -y \
	autoconf \
	bc \
	build-essential \
	g++-8 \
	gcc-8 \
	clang-8 \
	lld-8 \
	gettext-base \
	gfortran-8 \
	iputils-ping \
	libbz2-dev \
	libc++-dev \
	libcgal-dev \
	libffi-dev \
	libfreetype6-dev \
	libhdf5-dev \
	libjpeg-dev \
	liblzma-dev \
	libncurses5-dev \
	libncursesw5-dev \
	libpng-dev \
	libreadline-dev \
	libssl-dev \
	libsqlite3-dev \
	libxml2-dev \
	libxslt-dev \
	locales \
	moreutils \
	openssl \
	python-openssl \
	rsync \
	scons \
	python3-pip \
	libopenblas-dev \
	&& apt-get clean


RUN sudo apt-get clean && apt-get autoremove

RUN python3 -m pip install numpy scipy pyserial bitstring smbus2 grpcio-tools Pillow matplotlib spatialmath-python

ENV TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

RUN python3 -m pip install --upgrade pip; python3 -m pip install aiohttp numpy=='1.19.4' scipy=='1.5.3' export "LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH"; python3 -m pip install --upgrade protobuf; python3 -m pip install --no-cache $TORCH_INSTALL

RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' && \
    wget http://packages.ros.org/ros.key -O - | sudo apt-key add - && \
    sudo apt-get update && \
    sudo apt-get install -y python3-catkin-tools

# TODO: Remove all darknet stuff
RUN mkdir -p darknet_ws/src && \
    cd darknet_ws/src && \
    git clone --recursive https://github.com/Tartan-AUV/darknet_ros_orin.git darknet_ros

RUN mkdir -p cv_bridge_ws/src && \
    cd cv_bridge_ws/src && \
    git clone --recursive https://github.com/ros-perception/vision_opencv.git && \
    cd vision_opencv && \
    git checkout noetic

SHELL ["/bin/bash", "-c"] 

RUN cd cv_bridge_ws && \
    source /opt/ros/noetic/setup.bash && \
    catkin config --install --install-space /opt/tauv/packages && \
    catkin build cv_bridge -DCMAKE_BUILD_TYPE=Release && \
    source /opt/tauv/packages/setup.bash

RUN cd darknet_ws && \
    source /opt/ros/noetic/setup.bash && \
    source /opt/tauv/packages/setup.bash && \
    catkin config --install --install-space /opt/tauv/packages && \
    catkin build darknet_ros -DCMAKE_BUILD_TYPE=Release && \
    source /opt/tauv/packages/setup.bash

RUN sudo apt-get update -y

RUN sudo apt-get install -y tmux vim

# RUN echo "deb https://repo.download.nvidia.com/jetson/ffmpeg main main" |  sudo tee -a /etc/apt/sources.list && \
# echo "deb-src https://repo.download.nvidia.com/jetson/ffmpeg main main" |  sudo tee -a /etc/apt/sources.list && \
# sudo apt-get update -y && \
# sudo apt-get install -y -o DPkg::options::="--force-overwrite" ffmpeg

RUN sudo apt-mark hold libopencv libopencv-core4.2 libopencv-dev && \ 
    sudo apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-alsa \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav

RUN sudo apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev && \
    sudo apt-mark unhold libopencv libopencv-core4.2 libopencv-dev

RUN sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev && \
  git clone --branch release/0.15 https://github.com/pytorch/vision torchvision && \
  cd torchvision && \
  export BUILD_VERSION=0.15.0 && \
  python3 setup.py install --user

RUN echo 'source /opt/ros/noetic/setup.bash' >> /root/.bashrc
RUN echo 'source /opt/tauv/packages/setup.bash' >> /root/.bashrc
RUN echo 'source /shared/tauv_ws/devel/setup.bash' >> /root/.bashrc

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
WORKDIR /