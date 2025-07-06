#!/bin/bash

apt update
apt install pipx
pipx ensurepath
pip install tensorboard
pipx install poetry

cd workspace/
git clone https://github.com/nik3348/deepdreamer.git
cd deepdreamer
poetry install
