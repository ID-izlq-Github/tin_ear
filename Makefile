# 解释器优先级：项目内 .venv → 作者的 conda 环境 tin_ear → python3 → python；可用 PY= 覆盖
PY ?= $(shell for c in "$(CURDIR)/.venv/bin/python" "$(HOME)/miniforge3/envs/tin_ear/bin/python" python3 python; do if [ -x "$$c" ] || command -v "$$c" >/dev/null 2>&1; then echo "$$c"; break; fi; done)
HOST ?= 0.0.0.0
PORT ?= 8000
PROJ := $(CURDIR)
PID := $(PROJ)/data/server.pid
LOG := $(PROJ)/data/server.log

# 临时文件落在项目内：pytest 的合成音频与 ffmpeg 中间文件动辄几百 MB，
# 放 data/tmp 一来看得见、二来 make clean 能一次收走（运行时另有 app.py 的 tempfile.tempdir 兜底）。
# pip 下载缓存不再重定向 —— 用 pip 自己的默认位置（要清就 python -m pip cache purge）。
export TMPDIR := $(PROJ)/data/tmp

.PHONY: help install install-dev run run-bg stop test clean-cache clean size

help:
	@echo "make install     安装运行期依赖（解释器自动挑选，见下方；可用 PY= 指定）"
	@echo "make install-dev 额外安装测试依赖"
	@echo "make run         前台启动 http://localhost:$(PORT)（Ctrl+C 停止）"
	@echo "make run-bg      后台启动，日志 data/server.log"
	@echo "make stop        停止后台启动的实例"
	@echo "make test        跑自动化测试（约 1 分钟）"
	@echo ""
	@echo "清理（data/library/ 里的音频永远不动）："
	@echo "make clean-cache     清空测试音频缓存 data/cache（下次测试重新生成，短模式约 8 秒）"
	@echo "make clean           上面一项 + data/session、data/tmp、__pycache__、.pytest_cache"
	@echo "make size            看看各自占了多少"
	@echo ""
	@echo "pip 下载缓存用 pip 默认位置（不写进项目）；要清用 python -m pip cache purge"
	@echo ""
	@echo "当前解释器：$(PY)"
	@echo "可覆盖的变量：PY=/path/to/python  HOST=127.0.0.1  PORT=8000"

install:
	@mkdir -p "$(TMPDIR)"
	@echo "使用解释器：$(PY)"
	$(PY) -m pip install -r requirements.txt

install-dev: install
	$(PY) -m pip install -r requirements-dev.txt

run:
	@mkdir -p "$(TMPDIR)"
	@echo "使用解释器：$(PY)    地址：http://localhost:$(PORT)"
	$(PY) -m uvicorn app:app --host $(HOST) --port $(PORT) --workers 1

run-bg:
	@mkdir -p "$(PROJ)/data" "$(TMPDIR)"
	@echo "使用解释器：$(PY)"
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
	@echo "使用解释器：$(PY)"
	$(PY) -m pytest -q

clean-cache:
	rm -rf "$(PROJ)/data/cache"
	@echo "已清空测试音频缓存 data/cache（库内音频未动，下次测试会重新生成）"

# 可重建的一切；data/library/ 永远不动
clean:
	@if [ -f "$(PID)" ] && kill -0 "$$(cat $(PID))" 2>/dev/null; then \
		echo "⚠️  服务还在跑（pid $$(cat $(PID))），清理 data/session 会让进行中的测试失效；建议先 make stop"; \
	fi
	rm -rf "$(PROJ)/data/cache" "$(PROJ)/data/session" "$(PROJ)/data/tmp" \
	       "$(PROJ)/data/server.log" "$(PROJ)/data/server.pid" \
	       "$(PROJ)/.pytest_cache" "$(PROJ)/.pip-cache"
	find "$(PROJ)" -name __pycache__ -type d -prune -exec rm -rf {} +
	@mkdir -p "$(PROJ)/data/session" "$(PROJ)/data/tmp"
	@echo "已清理可重建的缓存（data/library 未动，仍有 $$(ls -1 "$(PROJ)/data/library"/*.flac 2>/dev/null | wc -l) 个音频文件）"

size:
	@du -sh "$(PROJ)/data/library" "$(PROJ)/data/cache" "$(PROJ)/data/tmp" \
	       "$(PROJ)/data/session" 2>/dev/null
	@du -sh "$(PROJ)/.pip-cache" 2>/dev/null || true  # 旧版本遗留的目录（现不再创建）
	@find "$(PROJ)" -name __pycache__ -type d -prune -exec du -sh {} + 2>/dev/null
	@echo "--- 合计 ---"
	@du -sh "$(PROJ)" 2>/dev/null
