#!/bin/bash

# General info about new container / image
CONTAINER_NAME=physicsnemo/norneressim
TAG=latest

# Build a container
docker build \
-t $CONTAINER_NAME .

