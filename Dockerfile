FROM continuumio/miniconda3

SHELL ["/bin/bash", "-c"]
ADD . /

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

USER 1001
CMD /opt/conda/envs/prophet-env/bin/python app.py
