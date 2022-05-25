FROM ubuntu:20.04 as decision_tree
RUN apt-get update && apt-get -y upgrade
ENV DEBIAN_FRONTEND=noninteractive

# install curl to allow pip and poetry installation
RUN apt-get install -y curl

# install python3.8
RUN apt-get install --no-install-recommends -y python3.8 python3.8-dev python3-distutils python3-venv
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 0

# install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

# install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# install graphviz onto the machine as it is required for the graphviz dependency
RUN apt-get install --no-install-recommends -y graphviz

# copy required files into docker image
WORKDIR /workdir
COPY pyproject.toml /workdir/pyproject.toml
COPY poetry.lock /workdir/poetry.lock
COPY Decision_Tree /workdir/Decision_Tree

# set poetry to not create a venv since docker image provides a suitable venv
RUN poetry config virtualenvs.create false
# setuptools is a python package, freeze setuptools due to bug which prevents poetry from working
RUN poetry run pip install "setuptools==59.8.0"

# install project dependencies
RUN poetry install

# set entry point for running docekr image
ENTRYPOINT ["decision-tree"]
