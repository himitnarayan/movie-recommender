#!/usr/bin/env bash

pip install -r requirements.txt

echo "Building ML model..."
python build_model.py

python manage.py collectstatic --noinput
