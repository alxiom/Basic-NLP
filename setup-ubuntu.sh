#!/bin/bash

cd
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
source ~/.bashrc
pyenv install miniconda3-4.7.10
pyenv global miniconda3-4.7.10
conda install -y pytorch==1.3.1
cd ~/multicampus-NLP
pip install -r requirements-ubuntu.txt 
mkdir data
wget https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt -O data/ratings.txt
wget https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData%20.csv -O data/chat.csv
sudo snap install pycharm-community --classic
