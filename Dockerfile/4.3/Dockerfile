FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.11

ARG JUPYTERHUB_VERSION=5.4.4

COPY ./requirements.txt .

RUN pip3 install --no-cache -r requirements.txt

RUN pip3 install --no-cache \
    jupyterhub==$JUPYTERHUB_VERSION

RUN pip3 install --no-cache \
    nbgitpuller
