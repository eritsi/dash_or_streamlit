#!/usr/bin/env bash

set -e

echo "=== Install Python packages ==="
pip install -r requirements.txt

echo "=== Install OS packages ==="
sudo apt-get update
sudo apt-get install -y graphviz poppler-utils unzip

echo "=== Configure Jupyter ==="
jupyter notebook --generate-config

{
  echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"'
  echo 'c.ContentsManager.default_jupytext_formats = "ipynb,py"'
  echo 'c.NotebookApp.password = ""'
} >> ~/.jupyter/jupyter_notebook_config.py

echo "=== Install Kaggle CLI ==="
pip install kaggle

echo "=== Download Kaggle Dataset ==="
# ここは Gitpod の "init" に相当だが、Codespaces では自動入力はできないため
# Kaggle API Key は Codespaces Secret で安全に扱う
if [[ -n "$KAGGLE_USERNAME" && -n "$KAGGLE_KEY" ]]; then
    kaggle d download -d salehahmedrony/gender-statistics -f Gender_StatsData.csv
    unzip Gender_StatsData.csv.zip
    rm Gender_StatsData.csv.zip
else
    echo "⚠️ Kaggle API credentials not set (KAGGLE_USERNAME / KAGGLE_KEY)"
    echo "Set them in GitHub → Repository → Settings → Secrets → Codespaces"
fi

echo "=== Setup complete ==="
