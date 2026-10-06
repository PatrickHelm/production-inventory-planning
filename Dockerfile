FROM python:3.9-slim

RUN mkdir /mnt/mydata/
WORKDIR /mnt/mydata/

# Copy the requirements file and install the dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
