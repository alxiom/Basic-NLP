#!/bin/bash

cd
sudo apt install -y git
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
pyenv install miniconda3-4.7.10
pyenv global miniconda3-4.7.10
conda install -y pytorch==1.3.1
pip install -r ~/multicampus-NLP/requirements-ubuntu.txt 
