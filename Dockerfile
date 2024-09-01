FROM nvidia/cuda:11.8.0-base-ubuntu22.04 as build-stage

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip git libopencv-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY dist/*.whl .
RUN python3 --version && pip3 install --upgrade pip && pip3 install *.whl && pip cache purge && rm *.whl

ENTRYPOINT ["/bin/sh"]
