# TAUV-MONO

This is the monorepo for all Tartan-AUV vehicle code, tools, and infrastructure. 

## Development Environment Setup(for sim and testing without the sub)

Follow these instructions to set up the development environment and simulator on a fresh machine.

### 1. Environment Setup
* **Create a Virtual Machine:** Install the newest version of Ubuntu.
    * *Disk Space:* **30GB** minimum (**40GB** recommended).
* **Enable OpenSSH:** Ensure you can access the VM remotely.
* **Install GUI:** Install XFCE (a lightweight GUI) to display the simulator later:
    ```bash
    sudo apt update
    sudo apt install xfce4
    ```
* **Connect via SSH:**
    1. Find the IP address of the VM:
       ```bash
       ip a
       ```
    2. Use this IP address to SSH into the machine for the remaining steps.

### 2. Install Dependencies
**Docker Engine**
1.  Install the Docker Engine following the official guide: [Install Docker on Ubuntu](https://docs.docker.com/engine/install/ubuntu/).
2.  (Optional) Add your user to the docker group to run docker without `sudo`:
    ```bash
    sudo usermod -aG docker $USER
    newgrp docker
    ```
    > `newgrp docker` activates the group in your current shell so you don't need to log out and back in.
3.  Authenticate with the GitHub Container Registry:
    ```bash
    docker login ghcr.io
    ```

**Git Configuration**
1.  Configure global Git settings:
    ```bash
    git config --global user.name "Your Name"
    git config --global user.email "your.email@example.com"
    ```
2.  Set up your SSH key with GitHub to allow for authentication during cloning.

### 3. Installation & Build
1.  **Clone the repository:** 
    ```bash
    git clone --recurse-submodules https://github.com/Tartan-AUV/tauv-mono.git
    ```

2.  **Build the Docker container:**
    ```bash
    cd tauv-mono/containers
    ./build_desktop_docker.sh
	docker compose up -d tauv-desktop
    ```

3. **Attach to the container:**
	Using the `Dev Containers` extension on VSCode, you should be able to attach to the container.
	* Your VSCode may complain about unsafe repositories. This is simply because our code is segmented into submodules, and you can just mark everything as safe.
    * Can also use `docker exec -it [Container Name] bash` to attach 

4.  **Build the ROS 2 Workspace:**
    Once inside the container, build the workspace and source the setup script **make sure you are in ros_ws folder** :
    ```bash
    colcon build --symlink-install
    source install/setup.bash
    ```
The development environment has now been built.


### 5. Useful Commands/Tips
**Docker**
* *To be added*

**ROS**
* *To be added*

## Jetson Setup

### 1. Connection
* SSH Via Tailscale or connecting to router direclty(we only have 3 seats total on tailnet unfortunately)
### 2. Build and Run the Container
1. Ensure you are in tauv-mono/containers and run `./build_osprey_docker.sh`
    * only needs to be done if a modification has been made to the dockerfiles 
2. Run The container 
    * If it is not running already `docker compose up -d osprey-orin-user'
    * Attach to it by `docker exec -it [Container name] bash`
3. ROS Stuff 
    * Will be in the `ros_ws` folder


# ROS Packages

## Dependencies

## Package list
Each package *will* contain a more detailed readme in their folder
