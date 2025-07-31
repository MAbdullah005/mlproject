FROM  python:3.13-slim

WORKDIR /application
COPY . .

RUN apt update -y && apt install awscli -y
RUN pip install -r requirements.txt
CMD [ "python3","application.py" ]
