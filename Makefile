all: bash


bash: build
	nvidia-docker-compose run bash

sunnybrook: build
	nvidia-docker-compose run train_sunnybrook

build:
	nvidia-docker-compose build
