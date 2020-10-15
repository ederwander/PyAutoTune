FROM python:3.7-alpine

COPY . /app
WORKDIR /app

# install numpy (https://gist.github.com/orenitamar/f29fb15db3b0d13178c1c4dd611adce2#gistcomment-3406770)
RUN apk update
RUN echo "http://dl-8.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories
RUN apk --no-cache --update-cache add gcc gfortran build-base wget freetype-dev libpng-dev openblas-dev
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
RUN pip install --no-cache-dir numpy

RUN python /app/setup.py install
