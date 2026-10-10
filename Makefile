PY := $(HOME)/miniforge3/envs/tin_ear/bin/python
PROJ := $(CURDIR)

# 环境纪律：所有缓存与临时文件留在项目目录内，不写系统/用户 temp 与 ~/.cache
export PIP_CACHE_DIR := $(PROJ)/.pip-cache
export TMPDIR := $(PROJ)/data/tmp

.PHONY: help install install-dev run test clean-cache

help:
	@echo "make install     安装运行期依赖到 conda 环境 tin_ear"
	@echo "make install-dev 额外安装测试依赖"
	@echo "make run         启动服务 http://localhost:8000"
	@echo "make test        跑自动化测试"
	@echo "make clean-cache 清空刺激缓存 data/cache"

install:
	@mkdir -p "$(PIP_CACHE_DIR)" "$(TMPDIR)"
	$(PY) -m pip install -r requirements.txt

install-dev: install
	$(PY) -m pip install -r requirements-dev.txt

run:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

test:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m pytest -q

clean-cache:
	rm -rf "$(PROJ)/data/cache"
