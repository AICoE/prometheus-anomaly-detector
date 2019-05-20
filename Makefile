ENV_FILE := .env
include ${ENV_FILE}
export $(shell sed 's/=.*//' ${ENV_FILE})
export PIPENV_DOTENV_LOCATION=${ENV_FILE}


run_app:
	pipenv run python app.py

run_app_local:
	python app.py
