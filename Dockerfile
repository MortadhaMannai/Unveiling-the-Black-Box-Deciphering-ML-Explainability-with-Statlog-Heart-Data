FROM ubuntu:18.04
RUN apt-get -yqq update
RUN apt-get install -yqq python3
RUN apt-get install -yqq python3-pip

COPY requirements.txt /app/

RUN pip3 install -r app/requirements.txt

COPY notebooks /app/notebooks
COPY data /app/data

WORKDIR /app


EXPOSE 8888

CMD ["jupyter", "notebook", "notebooks", "--port=8888", "--allow-root", "--no-browser", "--ip=0.0.0.0"]