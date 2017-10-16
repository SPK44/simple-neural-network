#!/bin/bash

docker run -it --rm -v $(pwd):/home/jovyan/work -p 8888:8888 jupyter/tensorflow-notebook 
