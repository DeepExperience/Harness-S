FROM slimerl/slime@sha256:939dc46f050fcb6dcaef9a276d5b08964fa4fb089539a59d96a450c3313fdb87
WORKDIR /opt/harness-s
COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends docker.io && rm -rf /var/lib/apt/lists/*
RUN pip install -e '.[runtime]' && pip install ray==2.55.1 transformers==5.3.0
ENV PYTHONUNBUFFERED=1

RUN ln -s "$(command -v python)" /opt/harness-s-python
