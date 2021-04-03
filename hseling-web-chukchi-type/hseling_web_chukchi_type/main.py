import os
from base64 import b64decode, b64encode
from flask import Flask, Blueprint, render_template, request, redirect, jsonify
from logging import getLogger

import requests


app = Flask(__name__)
app.config['DEBUG'] = False
app.config['LOG_DIR'] = '/tmp/'
if os.environ.get('HSELING_WEB_CHUKCHI_TYPE_SETTINGS'):
    app.config.from_envvar('HSELING_WEB_CHUKCHI_TYPE_SETTINGS')

app.config['HSELING_API_ENDPOINT'] = os.environ.get('HSELING_API_ENDPOINT')
app.config['HSELING_RPC_ENDPOINT'] = os.environ.get('HSELING_RPC_ENDPOINT')

print(app.config)


def get_server_endpoint():

    HSELING_API_ENDPOINT = app.config.get('HSELING_API_ENDPOINT')
    return HSELING_API_ENDPOINT


if not app.debug:
    import logging
    from logging.handlers import TimedRotatingFileHandler
    # https://docs.python.org/3.6/library/logging.handlers.html#timedrotatingfilehandler
    file_handler = TimedRotatingFileHandler(os.path.join(app.config['LOG_DIR'], 'hseling_web_chukchi_type.log'), 'midnight')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter('<%(asctime)s> <%(levelname)s> %(message)s'))
    app.logger.addHandler(file_handler)


log = getLogger(__name__)


@app.route('/web/healthz')
def healthz():
    app.logger.info('Health checked')
    return jsonify({"status": "ok", "message": "hseling-web-chukchi-type"})


@app.route('/web/')
def index():

    api_endpoint = get_server_endpoint()
    result = requests.get(api_endpoint).content

    return render_template('index.html.j2', result=result)


@app.route('/web/test')
def index_test():
    return render_template('index.html.j2', result="This is a string!")


@app.route('/')
def index_redirect():
    return redirect('/web/')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8000)


__all__ = [app]
