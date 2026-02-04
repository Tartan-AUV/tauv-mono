# TAUV-docker
docker configs for tartan auv

## Buidling and Running the Container on the ORIN

You should be in this directory for it all to work

```bash
./build_osprey_docker.sh # this will build the container
docker compose up -d osprey-orin-user # will spin up the onteiner
docker exec -it containers-osprey-orin-user-1 bash # Will attach to the running container can use ps to get container name (-it is interactive terminal) \

docker ps # will list out all the containers running
```
For how to run the ROS take a look at `ros_ws` README  TODO: ADD LINK HERE 

