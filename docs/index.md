# AMBER: Automated annotation and Multimodal Bag Extraction for Robotics

Amber is a ROS2 friendly ML tools.
Your rosbag2 become dataset!

## How it works

1. Prepare rosbag2 with mcap format.
2. Prepare task description yaml file.
3. Enjyo your ML life with Robots!

## How to setup?

### Check your OS is suppoted platform?
This tool is only support ubuntu 22.04.
Please install ubuntu 22.04 in your local machine first.

### Install Dependencies
#### Poetry
Setup environment and dependencies in python.
Please follow [this documentation.](https://python-poetry.org/docs/)

!!! warning
    Developer and github actions are tested under poetry 1.5.1

#### Docker
Some automation tools are executed inside docker.
Please follow [this documentation.](https://docs.docker.com/engine/install/ubuntu/)

!!! notion
    Developer use docker version 23.0.5

#### Nvidia driver and nvidia docker(Optional)
Some automation tools support cuda.
If you want to use gpu, please install nvidia driver and nvidia docker.

#### Google test(Optional)
Google test is a used for testing C++ code inside amber.
It is optional and it is not required for building amber in your local machine.

```bash
sudo apt update & sudo apt install -y googletest
```
