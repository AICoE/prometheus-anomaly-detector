# prometheus-anomaly-detector [(legacy-version)](https://github.com/AICoE/prometheus-anomaly-detector-legacy)
A framework to deploy a metric prediction model to detect anomalies in prometheus metrics.



## Feature Flags/Parameters:
* `FLT_PROM_URL` - URL for the prometheus host, from where the metric data will be collected
* `FLT_PROM_ACCESS_TOKEN` - OAuth token to be passed as a header, to connect to the prometheus host (Optional)
* `FLT_METRICS_LIST` - List of metrics that are to be collected from prometheus and train the prophet model.
<br> Example: `"up{app='openshift-web-console', instance='172.44.0.18:8443'}; up{app='openshift-web-console', instance='172.44.4.18:8443'}; es_process_cpu_percent{instance='172.44.17.134:30290'}"`, multiple metrics can be separated using a semi-colon `;`.
<br>If one metric and label configuration matches more than one timeseries, all the timeseries matching the configuration will be collected.
* `FLT_RETRAINING_INTERVAL_MINUTES` - This specifies the frequency of the model training, or how often the model is retrained. (Default: `15`)
<br> Example: If this parameter is set to `15`, it will collect the past 15 minutes of metric data every 15 minutes and append it to the training dataframe.
* `FLT_ROLLING_DATA_WINDOW_SIZE` - This parameter limits the size of the training dataframe to prevent Out of Memory errors. It can be set to the duration of data that should be stored in memory as dataframes. (Default `15d`)
<br> Example: If set to `1d`, every time before training the model using the training dataframe, the metric data that is older than 1 day will be deleted.
