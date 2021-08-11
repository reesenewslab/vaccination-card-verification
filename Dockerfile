From python:3

ENV PYTHONBUFFERED 1

WORKDIR /app

ADD . /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY . /app