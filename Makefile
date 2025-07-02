# Wintermute Makefile - Dev Environment Builder

# Environment Setup
PYTHON_VERSION=3.11.9
VENV_DIR=$(HOME)/wintermute/venv
VLLM_DIR=$(HOME)/wintermute/vllm
MODEL_NAME=NousResearch/Nous-Hermes-2-Mistral-7B-DPO
MODEL_BASE_NAME=$(shell basename $(MODEL_NAME))
MODEL_DIR=$(HOME)/wintermute/models/$(MODEL_BASE_NAME)
LOG_FILE=$(HOME)/wintermute/wintermute_setup.log

.PHONY: all python venv vllm model launch clean fix-ssl

all: python venv vllm model launch

build_no_launch: python venv vllm model

python:
	@echo "🐍 Installing Python $(PYTHON_VERSION)..."
	sudo apt update && sudo apt install -y \
	  build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
	  libnss3-dev libssl-dev libreadline-dev libffi-dev curl \
	  libsqlite3-dev wget git libbz2-dev liblzma-dev
	cd /tmp && \
	wget https://www.python.org/ftp/python/$(PYTHON_VERSION)/Python-$(PYTHON_VERSION).tgz && \
	tar -xf Python-$(PYTHON_VERSION).tgz && \
	cd Python-$(PYTHON_VERSION) && \
	./configure --enable-optimizations --with-openssl=/usr && \
	make -j$$(nproc) && \
	sudo make altinstall

venv:
	@echo "📁 Creating virtual environment..."
	/usr/local/bin/python3.10 -m venv $(VENV_DIR)
	bash -c "source $(VENV_DIR)/bin/activate && pip install --upgrade pip"

vllm:
	@echo "📦 Cloning and installing vLLM..."
	git clone https://github.com/vllm-project/vllm.git $(VLLM_DIR) || echo "Repo exists. Skipping."
	bash -c "source $(VENV_DIR)/bin/activate && cd $(VLLM_DIR) && pip install -e '.[dev,torch]'"

model:
	@echo "📥 Downloading model: $(MODEL_NAME)..."
	bash -c "source $(VENV_DIR)/bin/activate && pip install huggingface_hub"
	if [ ! -f "$(HOME)/.huggingface/token" ]; then \
		echo "🔐 Please login to HuggingFace CLI."; \
		huggingface-cli login; \
	fi
	huggingface-cli download $(MODEL_NAME) \
		--local-dir $(MODEL_DIR) --local-dir-use-symlinks False

launch:
	@echo "🚀 Launching vLLM server on port 8000..."
	bash -c "source $(VENV_DIR)/bin/activate && cd $(VLLM_DIR) && \
	python -m vllm.entrypoints.openai.api_server --model $(MODEL_DIR) --port 8000 >> $(LOG_FILE) 2>&1 &"
	sleep 5
	@echo "🔍 Checking vLLM API health..."
	curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "mistral", "prompt": "ping", "max_tokens": 1}' | grep -q 200 || \
		( echo "❌ API failed to respond properly." && exit 1 )
	@echo "✅ API responded successfully. Logs are in $(LOG_FILE)"

clean:
	@echo "🧹 Cleaning Wintermute build environment..."
	rm -rf $(VENV_DIR) $(VLLM_DIR) $(MODEL_DIR) $(LOG_FILE)

# Install Python dependencies
install:
	pip install -r talkingHead/backend/requirements.txt

# Run FastAPI backend with live reload
backend:
	cd talkingHead/backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


# Run Vite frontend
frontend:
	cd talkingHead/frontend && npm run dev


# Run both frontend and backend
dev:
	make -j2 backend frontend
