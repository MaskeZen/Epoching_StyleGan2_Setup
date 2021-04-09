#!/bin/bash

container_id=c1f1b9611d70

nvidia-docker container start $container_id
nvidia-docker exec -u $(id -u):$(id -g) -it $container_id bash