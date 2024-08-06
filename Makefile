# Makefile
# IMPORTANT: Please find the right cuda dev container for your environment
SHELL := /bin/bash

mmdet_install:
	python3 -m pip install pip --upgrade
	python3 -m pip install "torch==1.13.1"
	python3 -m pip install "torchvision==0.15.1"
	python3 -m pip install -U openmim
	python3 -m pip install "mmdet==3.0.0"
