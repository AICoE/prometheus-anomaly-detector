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
	
run_app:
	pipenv run python app.py

run_app_local:
	python app.py
