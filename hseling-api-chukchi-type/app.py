import sys
import logging
import os
import pathlib
import re
import waitress
from time import time_ns
from typing import List
from flask import abort, Blueprint, Flask, jsonify, Response, request
from flask_cors import CORS
from segmentation.ncrffpp import NCRFpp
from language_modelling.awdlstmlm.generate import Generator

HOST = '127.0.0.1:5000'
PYTHON = os.environ.get('WEB_DEVELOPMENT_PROJECT_PYTHON') or '/usr/bin/python3'
_ROOT = pathlib.Path(__file__).parent.resolve()
ROOT = str(_ROOT)
UPLOADS = _ROOT / 'models' / 'ncrfpp' / 'corpus_home'
AWD_LSTM = _ROOT / 'models' / 'awdlstm'
sys.path.append(str(_ROOT / 'language_modelling' / 'awdlstmlm'))
if not UPLOADS.exists():
    UPLOADS.mkdir()

app = Flask(__name__)
CORS(app)
myNCRFpp = NCRFpp(_ROOT / "models/ncrfpp/corpus_home", "ru_standard_v4", "models/ncrfpp/results", 10)
api = Blueprint('api', __name__)


def tokenize(input_text: str) -> List[str]:
    curr_time = time_ns()
    file_name = str(curr_time)
    path = UPLOADS / file_name
    with open(path, 'w') as f:
        f.write(input_text)

    raw_file_name = f"raw_{curr_time}"
    myNCRFpp.make_raw(file_name, raw_file_name)
    decode_file_path = f"results_{curr_time}"
    decode_config_path = f"config_{curr_time}"
    myNCRFpp.load_model("model.571.model", "model.dset", decode_file_path,
                        decode_config_path, raw_file_name)
    myNCRFpp.decode(PYTHON, ROOT, decode_config_path)
    res_file_name = f"res_{curr_time}"
    myNCRFpp.convert_bmes_to_words(decode_file_path, res_file_name)
    results = myNCRFpp.convert_words_to_strings(file_name, res_file_name)
    myNCRFpp.delete_corpus_files(file_name, raw_file_name, res_file_name)
    myNCRFpp.delete_results_files(decode_config_path, decode_file_path)
    return re.split(r"[ >]", results[0])


def get_prediction(input_text: str) -> str:
    tokens = tokenize(input_text)
    g = Generator(model_path=AWD_LSTM / 'chukchi_model.pt',
                  corpus_path=AWD_LSTM / 'chukchi_segments',
                  cuda=False)
    try:
        result = g.generate(','.join(tokens), 5)
    except Exception:
        raise Exception('Генерация не отработала')
    return result


@api.after_request
def log_response(response: Response) -> Response:
    logger = logging.getLogger('waitress')
    info = ' {host} {path} - {method} - {status}'.format(
        host=request.host, method=request.method,
        path=request.path, status=response.status)
    logger.info(info)
    return response


@api.route('/health')
def health():
    return jsonify({'success': 'story'})


@api.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    logger = logging.getLogger('waitress')
    input_text = request.json.get('text')
    if not input_text:
        abort(405, 'В запросе нет поля `text` или оно пустое')
    logger.info(' Запрос на генерацию принят в обработку')
    result = get_prediction(input_text)
    return jsonify({'suggestions': result.split(',')})


app.register_blueprint(api, url_prefix='/api')


if __name__ == '__main__':
    logging.getLogger('waitress').setLevel(logging.INFO)
    waitress.serve(app, listen=HOST, threads=1)
