#!/bin/bash

apt update
apt install -y pipx
pipx ensurepath
pip install tensorboard
pipx install poetry

cd workspace/
git clone https://github.com/nik3348/deepdreamer.git
cd deepdreamer
poetry install
tensorboard --logdir runs --host 0.0.0.0
