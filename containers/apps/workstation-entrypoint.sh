#!/bin/bash

# Start the SSH server in the background
/usr/sbin/sshd &

# Wait for SSH server to start properly
sleep 1

# Switch to the specified user, change to the specific directory, and start a Bash shell
exec su - $_USER -c "cd /home/$_USER && bash"
