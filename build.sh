#!/usr/bin/env bash

pip install -r requirements.txt
pip install kaggle

echo "Setting up Kaggle credentials..."
mkdir -p ~/.kaggle
echo "$KAGGLE_JSON" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo "Downloading TMDB dataset from Kaggle..."
mkdir -p recommender/data
kaggle datasets download -d asaniczka/tmdb-movies-dataset-2023-930k-movies -p recommender/data --unzip

echo "Building ML model..."
python build_model.py

python manage.py collectstatic --noinput
