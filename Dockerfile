FROM continuumio/miniconda3

RUN cat /etc/apt/sources.list
RUN sed --in-place --regexp-extended 's|http://|https://|g' /etc/apt/sources.list
RUN cat /etc/apt/sources.list
RUN apt update -qy && apt-get install -y make gcc g++ libstdc++6


SHELL ["/bin/bash", "-exo", "pipefail", "-c"]

#TODO do not use root in docker container
# RUN useradd -ms /bin/bash t4c
# USER t4c
# WORKDIR /home/t4c


ADD environment.yaml .
ADD install-requirements.txt .
ADD install-extras-torch-geometric.txt .
ADD t4c22/misc/check_torch_geometric_setup.py .
# https://docs.anaconda.com/anaconda/install/silent-mode/
RUN eval "$(/opt/conda/bin/conda shell.bash hook)" && \
    conda init bash && \
    source ~/.bashrc && \
    printenv && \
    export CONDA_ENVS_PATH=$PWD && \
    conda env update -f environment.yaml && \
    conda env list && \
    conda activate t4c22 && \
    python -m pip install -r install-extras-torch-geometric.txt -f https://data.pyg.org/whl/torch-1.11.0+cpu.html && \
    python --version && \
    python -c 'import torch_geometric' && \
    python check_torch_geometric_setup.py
