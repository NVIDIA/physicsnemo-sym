#!/bin/bash

# General info about new container / image
CONTAINER_NAME=physicsnemo/norneressim
CONTAINER_SHORTCUT_NAME=norneressim
SUBDIR_NAME=project
PORT_HOST=8890
TAG=latest

# Start container. Note that sudo is not necessary, but 
# the user won't have root rights anyway
# --rm \

#xhost +local:root

docker run \
    -it \
    --rm \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --runtime nvidia \
    --gpus all \
    -p ${PORT_HOST}:8888 \
    -v ${PWD}:/workspace/${SUBDIR_NAME} \
    --name ${CONTAINER_SHORTCUT_NAME} \
    ${CONTAINER_NAME}:${TAG} bash
