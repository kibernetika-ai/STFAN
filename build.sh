#!/usr/bin/env bash

docker build --tag kuberlab/stfan:cpu -f Dockerfile .
docker build --tag kuberlab/stfan:gpu -f Dockerfile.gpu .

if [ "$1" == "--push" ];
then
    docker push kuberlab/stfan:cpu
    docker push kuberlab/stfan:gpu
fi
