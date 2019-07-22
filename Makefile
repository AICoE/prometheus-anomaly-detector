ENV_FILE := .env
include ${ENV_FILE}
export $(shell sed 's/=.*//' ${ENV_FILE})
export PIPENV_DOTENV_LOCATION=${ENV_FILE}

oc_deploy_app:
	oc process --filename=openshift/oc-deployment-template.yaml \
		--param FLT_PROM_URL=${FLT_PROM_URL} \
		--param FLT_PROM_ACCESS_TOKEN=${FLT_PROM_ACCESS_TOKEN} \
		--param FLT_METRICS_LIST="${FLT_METRICS_LIST}" \
		--param FLT_DEBUG_MODE="${FLT_DEBUG_MODE}" \
		| oc apply -f -

oc_delete_app:
	oc delete all -l app=prometheus-anomaly-detector

oc_build_model_test_image:
	oc process --filename=openshift/oc-model-test-image-build-template.yaml \
		--param APPLICATION_NAME=${APPLICATION_NAME}-image \
		| oc apply -f -

oc_delete_model_test_image:
	oc process --filename=openshift/oc-model-test-image-build-template.yaml \
		--param APPLICATION_NAME=${APPLICATION_NAME}-image \
		| oc delete -f -

oc_run_model_test:
	oc process --filename=openshift/oc-model-test-job-template.yaml \
		--param NB_USER="`oc whoami`" \
		--param APPLICATION_NAME="${APPLICATION_NAME}-run" \
		--param MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
		--param APP_IMAGE_URI="${APPLICATION_NAME}-image" \
		--param FLT_PROM_URL=${FLT_PROM_URL} \
		--param FLT_PROM_ACCESS_TOKEN=${FLT_PROM_ACCESS_TOKEN} \
		--param FLT_METRICS_LIST="${FLT_METRICS_LIST}" \
		--param FLT_ROLLING_DATA_WINDOW_SIZE="${FLT_ROLLING_DATA_WINDOW_SIZE}" \
		--param FLT_RETRAINING_INTERVAL_MINUTES="${FLT_RETRAINING_INTERVAL_MINUTES}" \
		--param FLT_DEBUG_MODE="${FLT_DEBUG_MODE}" \
		| oc apply -f -

oc_delete_model_tests:
	oc process --filename=openshift/oc-model-test-job-template.yaml \
		--param NB_USER="`oc whoami`" \
		--param APPLICATION_NAME="${APPLICATION_NAME}-run" \
		--param MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
		--param APP_IMAGE_URI="${APPLICATION_NAME}-image" \
		--param FLT_PROM_URL=${FLT_PROM_URL} \
		--param FLT_PROM_ACCESS_TOKEN=${FLT_PROM_ACCESS_TOKEN} \
		--param FLT_METRICS_LIST="${FLT_METRICS_LIST}" \
		--param FLT_ROLLING_DATA_WINDOW_SIZE="${FLT_ROLLING_DATA_WINDOW_SIZE}" \
		--param FLT_RETRAINING_INTERVAL_MINUTES="${FLT_RETRAINING_INTERVAL_MINUTES}" \
		--param FLT_DEBUG_MODE="${FLT_DEBUG_MODE}" \
		| oc delete -f -

run_app_pipenv:
	pipenv run python app.py

run_test_pipenv:
	pipenv run python test_model.py

run_app:
	python app.py

run_test:
	python test_model.py
