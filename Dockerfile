FROM continuumio/miniconda3

SHELL ["/bin/bash", "-c"]
ADD . /
RUN chmod +x set_uid.sh

ADD environment.yml /tmp/environment.yml
RUN chmod g+w /etc/passwd
RUN conda env create -f /tmp/environment.yml

USER 1001

# Ensure that assigned uid has entry in /etc/passwd.
CMD ./set_uid.sh && /opt/conda/envs/prophet-env/bin/python ${APP_FILE}
