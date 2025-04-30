FROM gcc:12

RUN apt-get update && apt-get install -y ninja-build cmake


