# 베이스 이미지
FROM ubuntu:22.04

# Set the timezone to Asia/Seoul (KST)
RUN echo "tzdata tzdata/Areas select Asia" | debconf-set-selections && \
    echo "tzdata tzdata/Zones/Asia select Seoul" | debconf-set-selections && \
    ln -snf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && echo "Asia/Seoul" > /etc/timezone

ENV DEBIAN_FRONTEND="noninteractive"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# essential elements
RUN set -ex && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends \
        ca-certificates \
        git \
        sudo \
        gdb \
        vim \
        curl \
        wget \
        gnupg2 \
        dirmngr \
        software-properties-common \
        cmake \
        build-essential \
        python3 \
        python3-pip \
        python3-tk \
        # GUI
        x11-xserver-utils \
        libgl1-mesa-glx \
        libgl1-mesa-dri \
        xvfb \
        xauth && \
    update-ca-certificates

RUN apt-get -y clean && \ 
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# add user
ARG USERNAME=infra
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} ${USERNAME} && \
    useradd \
        -N \
        -m \
        -u ${USER_ID} \
        -g ${USERNAME} \
        -G "sudo,dialout" \
        -s "/bin/bash" ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    echo "${USERNAME}:1234" | chpasswd && \
    mkdir -p /run/user/${USER_ID} && \
    chown -R ${USERNAME}:${USERNAME} /run/user/${USER_ID} && \
    chmod 700 /run/user/${USER_ID} && \
    usermod -aG video ${USERNAME}