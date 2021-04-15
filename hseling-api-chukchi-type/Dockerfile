FROM python:3.8
WORKDIR /usr/src/app
COPY hseling-api-chukchi-type/requirements.txt ./
RUN pip install -r requirements.txt
COPY hseling-api-chukchi-type/. .
EXPOSE 5000
CMD ["python","app.py"]
