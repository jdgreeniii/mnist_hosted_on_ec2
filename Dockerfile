FROM python:3.6-slim
RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get clean
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT [ "python", "app.py" ]