#!/bin/bash
# Image Info version: pytorch_2.1.0-cann_8.0.rc2-py_3.9-euler_2.10.7-aarch64-snt9b

set -e  # 遇到错误立即退出

# 设置pip为非交互模式
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# 升级pip
pip install --upgrade pip --quiet --no-warn-script-location

# 批量安装包
packages=(
    "torchdata==0.6.1 --no-deps"
    "toolz==1.0.0 --no-deps"
    "pydantic==2.7.1 --no-deps"
    "pydantic-core==2.18.2 --no-deps"
    "typing-inspection --no-deps"
    "annotated-types==0.6.0 --no-deps"
    "docker-pycreds==0.4.0 --no-deps"
    "sentry-sdk==2.15.0 --no-deps"
    "wandb==0.19.4 --no-deps"
    "pyarrow==16.0.0"
    "numpy==1.23.5"
    "anndata==0.10.9"
    "pandas==1.5.3"
    "dask[complete]==2023.12.1"
    "scanpy==1.10.1"
    "transformers==4.40.2"
)

# 循环安装
for package in "${packages[@]}"; do
    echo "Installing: $package"
    pip install $package --no-input --exists-action i
done

echo "所有包安装完成！"
