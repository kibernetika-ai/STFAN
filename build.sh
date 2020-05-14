#!/usr/bin/env bash

docker build --tag kuberlab/stfan:gpu -f Dockerfile .

if [ "$1" == "--push" ];
then
    docker push kuberlab/stfan:gpu
fi
