FROM python:3.8-buster
USER root
RUN apt update
ENV PYTHONUNBUFFERED 1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# add kedro user
# ARG KEDRO_UID=999
# ARG KEDRO_GID=0
# RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
# useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} root
# USER kedro_docker
WORKDIR /home/kedro_docker

# COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .
COPY . .
