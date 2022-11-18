FROM python:3.10-slim-bullseye

RUN apt update -y

RUN pip3 --no-cache-dir install --upgrade awscli

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

CMD ["python3", "app.py"]