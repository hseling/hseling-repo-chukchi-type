import os
from base64 import b64decode, b64encode
from flask import Flask, jsonify, request
from logging import getLogger



from hseling_api_chukchi_type import boilerplate

from hseling_lib_chukchi_type.process import process_data
from hseling_lib_chukchi_type.query import query_data


app = Flask(__name__)
app.config['DEBUG'] = False
app.config['LOG_DIR'] = '/tmp/'
if os.environ.get('HSELING_API_CHUKCHI_TYPE_SETTINGS'):
    app.config.from_envvar('HSELING_API_CHUKCHI_TYPE_SETTINGS')



if not app.debug:
    import logging
    from logging.handlers import TimedRotatingFileHandler
    # https://docs.python.org/3.6/library/logging.handlers.html#timedrotatingfilehandler
    file_handler = TimedRotatingFileHandler(os.path.join(app.config['LOG_DIR'], 'hseling_api_chukchi_type.log'), 'midnight')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter('<%(asctime)s> <%(levelname)s> %(message)s'))
    app.logger.addHandler(file_handler)

log = getLogger(__name__)


ALLOWED_EXTENSIONS = ['txt']


def do_process_task(file_ids_list):
    files_to_process = boilerplate.list_files(recursive=True,
                                              prefix=boilerplate.UPLOAD_PREFIX)
    if file_ids_list:
        files_to_process = [boilerplate.UPLOAD_PREFIX + file_id
                            for file_id in file_ids_list
                            if (boilerplate.UPLOAD_PREFIX + file_id)
                            in files_to_process]
    data_to_process = {file_id[len(boilerplate.UPLOAD_PREFIX):]:
                       boilerplate.get_file(file_id)
                       for file_id in files_to_process}
    processed_file_ids = list()
    for processed_file_id, contents in process_data(data_to_process):
        processed_file_ids.append(
            boilerplate.add_processed_file(
                processed_file_id,
                contents,
                extension='txt'
            ))
    return processed_file_ids

@app.route('/api/healthz')
def healthz():
    app.logger.info('Health checked')
    return jsonify({"status": "ok", "message": "hseling-api-chukchi-type"})

@app.route('/api/upload', methods=['GET', 'POST'])
def upload_endpoint():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": boilerplate.ERROR_NO_FILE_PART})
        upload_file = request.files['file']
        if upload_file.filename == '':
            return jsonify({"error": boilerplate.ERROR_NO_SELECTED_FILE})
        if upload_file and boilerplate.allowed_file(
                upload_file.filename,
                allowed_extensions=ALLOWED_EXTENSIONS):
            return jsonify(boilerplate.save_file(upload_file))
    return boilerplate.get_upload_form()

@app.route('/api/files/<path:file_id>')
def get_file_endpoint(file_id):
    if file_id in boilerplate.list_files(recursive=True):
        return boilerplate.get_file(file_id)
    return jsonify({'error': boilerplate.ERROR_NO_SUCH_FILE})

@app.route('/api/files')
def list_files_endpoint():
    return jsonify({'file_ids': boilerplate.list_files(recursive=True)})

def do_process(file_ids):
    file_ids_list = file_ids and file_ids.split(",")
    result = do_process_task(file_ids_list)
    return {"result": result}

@app.route('/api/process')
@app.route("/api/process/<file_ids>")
def process_endpoint(file_ids=None):
    return jsonify(do_process(file_ids))


def do_query(file_id, query_type):
    if not query_type:
        return {"error": boilerplate.ERROR_NO_QUERY_TYPE_SPECIFIED}
    processed_file_id = boilerplate.PROCESSED_PREFIX + file_id
    if processed_file_id in boilerplate.list_files(recursive=True):
        return {
            "result": query_data({
                processed_file_id: boilerplate.get_file(processed_file_id)
            }, query_type=query_type)
        }
    return {"error": boilerplate.ERROR_NO_SUCH_FILE}

@app.route("/api/query/<path:file_id>")
def query_endpoint(file_id):
    query_type = request.args.get('type')
    return jsonify(do_query(file_id, query_type))

@app.route("/api/status/<task_id>")
def status_endpoint(task_id):
    return jsonify(boilerplate.get_task_status(task_id))



def get_endpoints(ctx):
    def endpoint(name, description, active=True):
        return {
            "name": name,
            "description": description,
            "active": active
        }

    all_endpoints = [
        endpoint("root", boilerplate.ENDPOINT_ROOT),
        endpoint("scrap", boilerplate.ENDPOINT_SCRAP,
                 not ctx["restricted_mode"]),
        endpoint("upload", boilerplate.ENDPOINT_UPLOAD),
        endpoint("process", boilerplate.ENDPOINT_PROCESS),
        endpoint("query", boilerplate.ENDPOINT_QUERY),
        endpoint("status", boilerplate.ENDPOINT_STATUS)
    ]

    return {ep["name"]: ep for ep in all_endpoints if ep}


@app.route("/api/")
def main_endpoint():
    ctx = {"restricted_mode": boilerplate.RESTRICTED_MODE}
    return jsonify({"endpoints": get_endpoints(ctx)})




if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)


__all__ = [app]
