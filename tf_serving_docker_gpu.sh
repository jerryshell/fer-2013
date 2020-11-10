#!/usr/bin/env bash
docker run --rm  -it -p 8501:8501 -v "$(pwd)/model_tf:/models/fer/0" -e MODEL_NAME=fer tensorflow/serving:latest-gpu
