FROM hseling/hseling-api-base:python3.6-alpine3.7 as hseling_web_build

LABEL maintainer="Mikhail Voronov <mikivo@list.ru>"

RUN mkdir /dependencies
COPY ./hseling-web-chukchi-type/requirements.txt /dependencies/requirements.txt
COPY ./hseling-web-chukchi-type/setup.py /dependencies/setup.py

RUN pip install -r /dependencies/requirements.txt

FROM hseling/hseling-api-base:python3.6-alpine3.7 as hseling_web_production

COPY --from=hseling_web_build /usr/local/lib/python3.6/site-packages /usr/local/lib/python3.6/site-packages
COPY --from=hseling_web_build /usr/lib/python3.6/site-packages /usr/lib/python3.6/site-packages

COPY --from=hseling_web_build /dependencies /dependencies

COPY ./hseling-web-chukchi-type/hseling_web_chukchi_type /app/hseling_web_chukchi_type
COPY ./hseling-web-chukchi-type/hseling_web_chukchi_type/static /app/static
COPY ./hseling-web-chukchi-type/main.py /app/main.py

ENV HSELING_WEB_CHUKCHI_TYPE_SETTINGS="/app/hseling_web_chukchi_type/default_settings.py"

HEALTHCHECK CMD curl --fail http://localhost:8000/web/healthz || exit 1

EXPOSE 8000
