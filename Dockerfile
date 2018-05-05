FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04
MAINTAINER Borodin Gregory <grihabor@mail.ru>

RUN apt update \
 && apt install -y python3-pip \
 && apt autoremove

RUN pip3 install \
        numpy==1.11 \
        opencv-python==3.1 \
        tensorflow-gpu==1.0.1 \
        keras==2.0.2 \
        h5py==2.7 \
        pydicom==0.9.9 \
        scikit-image==0.13

RUN apt install -y libglib2.0-dev

VOLUME /project
WORKDIR /project

CMD ["/bin/bash"]
