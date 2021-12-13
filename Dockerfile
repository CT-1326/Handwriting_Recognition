FROM tensorflow/tensorflow:2.5.0

WORKDIR /usr/src

COPY ./requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["Model/index.py"]
ENTRYPOINT ["python3"]