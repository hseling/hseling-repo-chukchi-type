all: run

clean:
	rm -rf venv && rm -rf *.egg-info && rm -rf dist && rm -rf *.log*

venv:
	virtualenv --python=python3 venv && venv/bin/pip install ../hseling_lib_chukchi_type && venv/bin/pip install .

run: venv
	FLASK_APP=hseling_api_chukchi_type venv/bin/flask run

test: venv
	HSELING_API_CHUKCHI_TYPE_SETTINGS=../settings.cfg venv/bin/python -m unittest discover -s tests

sdist: venv test
	venv/bin/python setup.py sdist
