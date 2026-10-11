# 解释器优先级：项目内 .venv → 作者的 conda 环境 tin_ear → python3 → python；可用 PY= 覆盖
PY ?= $(shell for c in "$(CURDIR)/.venv/bin/python" "$(HOME)/miniforge3/envs/tin_ear/bin/python" python3 python; do if [ -x "$$c" ] || command -v "$$c" >/dev/null 2>&1; then echo "$$c"; break; fi; done)
HOST ?= 0.0.0.0
PORT ?= 8000
PROJ := $(CURDIR)
PID := $(PROJ)/data/server.pid
LOG := $(PROJ)/data/server.log

# 环境纪律：所有缓存与临时文件留在项目目录内，不写系统/用户 temp 与 ~/.cache
export PIP_CACHE_DIR := $(PROJ)/.pip-cache
export TMPDIR := $(PROJ)/data/tmp

.PHONY: help install install-dev run run-bg stop test clean-cache

help:
	@echo "make install     安装运行期依赖（默认装进 conda 环境 tin_ear；可用 PY= 指定解释器）"
	@echo "make install-dev 额外安装测试依赖"
	@echo "make run         前台启动 http://localhost:$(PORT)（Ctrl+C 停止）"
	@echo "make run-bg      后台启动，日志 data/server.log"
	@echo "make stop        停止后台启动的实例"
	@echo "make test        跑自动化测试（约 1 分钟）"
	@echo "make clean-cache 清空片段与刺激缓存 data/cache"
	@echo ""
	@echo "当前解释器：$(PY)"
	@echo "可覆盖的变量：PY=/path/to/python  HOST=127.0.0.1  PORT=8000"

install:
	@mkdir -p "$(PIP_CACHE_DIR)" "$(TMPDIR)"
	$(PY) -m pip install -r requirements.txt

install-dev: install
	$(PY) -m pip install -r requirements-dev.txt

run:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m uvicorn app:app --host $(HOST) --port $(PORT) --workers 1

run-bg:
	@mkdir -p "$(PROJ)/data" "$(TMPDIR)"
	@$(PY) -m uvicorn app:app --host $(HOST) --port $(PORT) --workers 1 > "$(LOG)" 2>&1 & echo $$! > "$(PID)"
	@sleep 1
	@echo "已后台启动 (pid $$(cat $(PID)))：浏览器打开 http://localhost:$(PORT)"
	@echo "日志：$(LOG)    停止：make stop"

stop:
	@if [ -f "$(PID)" ] && kill -0 "$$(cat $(PID))" 2>/dev/null; then \
		kill "$$(cat $(PID))"; \
		for i in 1 2 3 4 5 6 7 8 9 10; do \
			kill -0 "$$(cat $(PID))" 2>/dev/null || break; sleep 0.5; \
		done; \
		rm -f "$(PID)"; \
		if command -v ss >/dev/null 2>&1; then busy=$$(ss -ltn 2>/dev/null | grep -c ":$(PORT)"); \
		elif command -v lsof >/dev/null 2>&1; then busy=$$(lsof -nP -iTCP:$(PORT) -sTCP:LISTEN 2>/dev/null | wc -l); \
		else busy=0; fi; \
		if [ "$$busy" != "0" ]; then echo "已发送停止信号，但 $(PORT) 端口仍被占用，请稍等或检查"; else echo "已停止，$(PORT) 端口已释放"; fi; \
	else rm -f "$(PID)"; echo "没有在运行的实例"; fi

test:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m pytest -q

clean-cache:
	rm -rf "$(PROJ)/data/cache"
