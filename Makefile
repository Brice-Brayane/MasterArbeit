VENV_PY=python
.PHONY: setup test test-hw lint run
setup:
	$(VENV_PY) -m pip install -U pip wheel
	$(VENV_PY) -m pip install -r requirements-dev.txt
test:
	$(VENV_PY) -m pytest -q -m "not hw"
test-hw:
	$(VENV_PY) -m pytest -q -m hw
lint:
	pre-commit run --all-files
run:
	$(VENV_PY) -m scripts.detect_realtime --model $$HOME/MasterArbeit/models/efficientdet-lite2.tflite --score 0.4 --threads 4 --width 1280 --height 720
