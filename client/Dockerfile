FROM python:3.8.11
WORKDIR ./
ADD request.py .
RUN apt-get update 
RUN apt-get install -y vim
RUN apt-get install -y iputils-ping
RUN apt-get install -y curl
RUN pip install requests

