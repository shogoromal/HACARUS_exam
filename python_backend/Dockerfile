FROM public.ecr.aws/docker/library/python:3.9

COPY . /python_backend
WORKDIR /python_backend

# SHELL ["/bin/bash", "-c"]
RUN apt-get update &
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt

#COPY ./src .

ENV PATH $PATH:usr/local/bin

WORKDIR /python_backend/src

CMD uvicorn main:app --reload --host 0.0.0.0 --port 8081