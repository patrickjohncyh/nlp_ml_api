FROM python:3.8 as base

WORKDIR /home/endpoint

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install git+https://github.com/patrickjohncyh/nlp_ml_api

WORKDIR /production

COPY . .

ARG PORT=80
ARG HOST=0.0.0.0
ARG APP_MODULE=endpoint.app:app
ARG WORKERS_PER_CORE=1

ENV MODE=production
ENV APP_MODULE=${APP_MODULE}
ENV WORKERS_PER_CORE=${WORKERS_PER_CORE}}
ENV HOST=${HOST}
ENV PORT=${PORT}

EXPOSE ${PORT}

ENTRYPOINT [ "./scripts/start.sh" ]
