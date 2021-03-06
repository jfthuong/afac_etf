
FROM gitpod/workspace-full

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN python3 -m pip install -r requirements.txt
COPY . /opt/app

USER gitpod


# Install custom tools, runtime, etc. using apt-get
# For example, the command below would install "bastet" - a command line tetris clone:
#
# RUN sudo apt-get -q update && #     sudo apt-get install -yq bastet && #     sudo rm -rf /var/lib/apt/lists/*
#
# More information: https://www.gitpod.io/docs/config-docker/
