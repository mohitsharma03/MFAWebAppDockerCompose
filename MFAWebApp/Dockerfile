FROM python:3.9-slim-buster

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0
#RUN pip3 install mysql-connector-python-rf

RUN pip3 install -r requirements.txt

#CMD ["python3", "Code/app.py"]

