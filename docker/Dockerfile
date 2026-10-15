FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    zip \
    unzip \
    libgl1-mesa-glx \
    git \
    xauth \
    x11-apps \
    python3-pyqt5

RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN pip3 install flask flask-cors gunicorn

WORKDIR /IVAS
COPY . /IVAS
RUN pip install -r requirements.txt

ENV DISPLAY=:0
RUN chmod +x /IVAS/docker/entrypoint.sh

ENTRYPOINT ["bash", "/IVAS/docker/entrypoint.sh"]



