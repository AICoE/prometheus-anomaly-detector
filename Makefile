ENV_FILE := .env
include ${ENV_FILE}
export $(shell sed 's/=.*//' ${ENV_FILE})
export PIPENV_DOTENV_LOCATION=${ENV_FILE}

oc_build_image:
	oc process --filename=openshift/oc-image-build-template.yaml \
		--param APPLICATION_NAME=${OC_APP_IMAGE_NAME} \
		| oc apply -f -

oc_trigger_build:
	oc start-build ${OC_APP_IMAGE_NAME}

oc_delete_image:
	oc process --filename=openshift/oc-image-build-template.yaml \
		--param APPLICATION_NAME=${OC_APP_IMAGE_NAME} \
		| oc delete -f -

oc_deploy_app:
	oc process --filename=openshift/oc-deployment-template.yaml \
		--param FLT_PROM_URL=${FLT_PROM_URL} \
		--param FLT_PROM_ACCESS_TOKEN=${FLT_PROM_ACCESS_TOKEN} \
		--param FLT_METRICS_LIST="${FLT_METRICS_LIST}" \
		--param FLT_DEBUG_MODE="${FLT_DEBUG_MODE}" \
		--param OC_APP_IMAGE_NAME="${OC_APP_IMAGE_NAME}" \
		| oc apply -f -

oc_delete_app:
	oc process --filename=openshift/oc-deployment-template.yaml \
	--param FLT_PROM_URL=${FLT_PROM_URL} \
	--param FLT_PROM_ACCESS_TOKEN=${FLT_PROM_ACCESS_TOKEN} \
	--param FLT_METRICS_LIST="${FLT_METRICS_LIST}" \
	--param FLT_DEBUG_MODE="${FLT_DEBUG_MODE}" \
	--param OC_APP_IMAGE_NAME="${OC_APP_IMAGE_NAME}" \
		| oc delete -f -

oc_run_model_test:
	oc process --filename=openshift/oc-model-test-job-template.yaml \
		--param APPLICATION_NAME="${APPLICATION_NAME}" \
		--param NB_USER="`oc whoami`" \
		--param MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
		--param OC_APP_IMAGE_NAME="${OC_APP_IMAGE_NAME}" \
		--param FLT_PROM_URL=${FLT_PROM_URL} \
		--param FLT_PROM_ACCESS_TOKEN=${FLT_PROM_ACCESS_TOKEN} \
		--param FLT_METRICS_LIST="${FLT_METRICS_LIST}" \
		--param FLT_DATA_START_TIME="${FLT_DATA_START_TIME}" \
		--param FLT_DATA_END_TIME="${FLT_DATA_END_TIME}" \
		--param FLT_ROLLING_TRAINING_WINDOW_SIZE="${FLT_ROLLING_TRAINING_WINDOW_SIZE}" \
		--param FLT_RETRAINING_INTERVAL_MINUTES="${FLT_RETRAINING_INTERVAL_MINUTES}" \
		--param FLT_DEBUG_MODE="${FLT_DEBUG_MODE}" \
		| oc apply -f -

oc_delete_model_test:
	oc process --filename=openshift/oc-model-test-job-template.yaml \
	--param APPLICATION_NAME="${APPLICATION_NAME}" \
	--param NB_USER="`oc whoami`" \
	--param MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
	--param OC_APP_IMAGE_NAME="${OC_APP_IMAGE_NAME}" \
	--param FLT_PROM_URL=${FLT_PROM_URL} \
	--param FLT_PROM_ACCESS_TOKEN=${FLT_PROM_ACCESS_TOKEN} \
	--param FLT_METRICS_LIST="${FLT_METRICS_LIST}" \
	--param FLT_DATA_START_TIME="${FLT_DATA_START_TIME}" \
	--param FLT_DATA_END_TIME="${FLT_DATA_END_TIME}" \
	--param FLT_ROLLING_TRAINING_WINDOW_SIZE="${FLT_ROLLING_TRAINING_WINDOW_SIZE}" \
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
