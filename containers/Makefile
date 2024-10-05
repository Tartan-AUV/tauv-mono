# builds the docker container
.PHONY: build
build:
	docker-compose build

.PHONY: build-packages
build-packages:
	docker-compose build --build-arg CACHEBUST=$(date +%s)

# Builds and starts the docker container in the background
.PHONY: up
up: build
	docker-compose up -d

# Kills the docker container
.PHONY: kill
kill:
	docker-compose kill && docker-compose rm

# Attaches a shell to the docker container
.PHONY: attach
attach:
	docker-compose exec tauvservice /bin/bash

.PHONY: rm
rm:
	docker-compose rm

# Same as up, but also recreates the docker-compose config. (reloads the yaml, basically)
.PHONY: recreate-up
recreate-up: build
	docker-compose up -d --force-recreate


.PHONY: install
install:
	docker run hello-world
	sudo apt-get install -y \
			ca-certificates \
			python3-pip \
			curl \
			gnupg \
			lsb-release
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
	echo \
		"deb [arch=$(shell dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
		focal stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
	sudo apt-get update
	sudo apt-get install -y docker-ce docker-ce-cli containerd.io
	sudo groupadd -f docker
	sudo usermod -aG docker $(USER)
	pip3 install docker-compose
	newgrp docker
