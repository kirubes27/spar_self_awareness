#!/bin/bash
gunicorn --workers 3 --bind 0.0.0.0:5000 tom_web_app:app