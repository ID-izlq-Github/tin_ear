PY := $(HOME)/miniforge3/envs/tin_ear/bin/python
PROJ := $(CURDIR)
PID := $(PROJ)/data/server.pid
LOG := $(PROJ)/data/server.log

# 环境纪律：所有缓存与临时文件留在项目目录内，不写系统/用户 temp 与 ~/.cache
export PIP_CACHE_DIR := $(PROJ)/.pip-cache
export TMPDIR := $(PROJ)/data/tmp

.PHONY: help install install-dev run run-bg stop test clean-cache

help:
	@echo "make install     把运行期依赖装进本项目专用 conda 环境 tin_ear"
	@echo "make install-dev 额外安装测试依赖"
	@echo "make run         前台启动 http://localhost:8000（Ctrl+C 停止）"
	@echo "make run-bg      后台启动，日志 data/server.log"
	@echo "make stop        停止后台启动的实例"
	@echo "make test        跑自动化测试（49 项，约 1 分钟）"
	@echo "make clean-cache 清空片段与刺激缓存 data/cache"

install:
	@mkdir -p "$(PIP_CACHE_DIR)" "$(TMPDIR)"
	$(PY) -m pip install -r requirements.txt

install-dev: install
	$(PY) -m pip install -r requirements-dev.txt

run:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

run-bg:
	@mkdir -p "$(PROJ)/data" "$(TMPDIR)"
	@$(PY) -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 > "$(LOG)" 2>&1 & echo $$! > "$(PID)"
	@sleep 1
	@echo "已后台启动 (pid $$(cat $(PID)))：Windows 浏览器打开 http://localhost:8000"
	@echo "日志：$(LOG)    停止：make stop"

stop:
	@if [ -f "$(PID)" ] && kill -0 "$$(cat $(PID))" 2>/dev/null; then \
		kill "$$(cat $(PID))"; \
		for i in 1 2 3 4 5 6 7 8 9 10; do \
			kill -0 "$$(cat $(PID))" 2>/dev/null || break; sleep 0.5; \
		done; \
		rm -f "$(PID)"; \
		if ss -ltn 2>/dev/null | grep -q ':8000'; then echo "已发送停止信号，但 8000 端口仍被占用，请稍等或检查"; else echo "已停止，8000 端口已释放"; fi; \
	else rm -f "$(PID)"; echo "没有在运行的实例"; fi

test:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m pytest -q

clean-cache:
	rm -rf "$(PROJ)/data/cache"
