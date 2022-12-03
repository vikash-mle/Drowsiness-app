#!/bin/bash

set -e

export IMAGE_NAME="frontend-app"

docker build -t $IMAGE_NAME -f Dockerfile .
docker run --rm --name $IMAGE_NAME -ti -v "$(pwd)/:/app/" -p 3000:3000 $IMAGE_NAME