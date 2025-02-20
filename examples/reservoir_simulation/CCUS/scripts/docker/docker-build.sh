#!/bin/bash

# General info about new container / image
CONTAINER_NAME=physicsnemo/ccusressim
TAG=latest

# Build a container
docker build \
-t $CONTAINER_NAME .

