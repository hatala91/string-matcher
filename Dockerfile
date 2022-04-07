# syntax=docker/dockerfile:experimental
FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc musl-dev git openssh-client && \
    pip3 install --upgrade poetry && \
    mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

COPY /matcher /matcher

WORKDIR /matcher

RUN poetry install
