FROM hseling/hseling-api-base:python3.6-alpine3.7 as hseling_api_build

LABEL maintainer="Mikhail Voronov <mikivo@list.ru>"

RUN mkdir /dependencies
COPY ./hseling-api-chukchi-type/requirements.txt /dependencies/requirements.txt
COPY ./hseling-api-chukchi-type/setup.py /dependencies/setup.py

RUN pip install -r /dependencies/requirements.txt

FROM hseling/hseling-api-base:python3.6-alpine3.7 as production

COPY --from=hseling_api_build /usr/local/lib/python3.6/site-packages /usr/local/lib/python3.6/site-packages
COPY --from=hseling_api_build /usr/lib/python3.6/site-packages /usr/lib/python3.6/site-packages

COPY --from=hseling_api_build /dependencies /dependencies

COPY ./hseling-lib-chukchi-type /dependencies/hseling-lib-chukchi-type
RUN pip install /dependencies/hseling-lib-chukchi-type

COPY ./hseling-api-chukchi-type/hseling_api_chukchi_type /app/hseling_api_chukchi_type
COPY ./hseling-api-chukchi-type/main.py /app/main.py

RUN mkdir -p /data

ENV HSELING_API_CHUKCHI_TYPE_SETTINGS="/app/hseling_api_chukchi_type/default_settings.py"

HEALTHCHECK CMD curl --fail http://localhost:5000/api/healthz || exit 1

