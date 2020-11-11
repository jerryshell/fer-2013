#!/usr/bin/env bash
docker run --name tf_serving_fer --gpus all -d -p 8585:8501 -v "$(pwd)/model_tf:/models/fer/0" -e MODEL_NAME=fer tensorflow/serving:latest-gpu
