# VARIABLES
ROOT:=./
VENV_BIN_DIR:="venv/bin"

PIP:="$(VENV_BIN_DIR)/pip"
LOCAL:="$(VENV_BIN_DIR)/streamlit"

VIRTUALENV:=$(shell which virtualenv)
REQUIREMENTS:="requirements.txt"

APP_NAME=wikipedia-nlp
DOCKER_IMAGE_REMOTE=vicotrbb/$(APP_NAME)

# PHONY

.PHONY: help clean venv test test-simple run docker-run docker-build up

# UTILS

help:
	@echo "#####################--HELP--#####################"
	@echo "#####################--COMMANDS--#####################"

clean:
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	@find . -type d -name .pytest_cache -delete
	@rm -rf venv

# DEVELOPMENT

define create-venv
virtualenv venv -p python3
endef

venv:
	$(create-venv)
	@$(PIP) install --no-cache-dir -r $(REQUIREMENTS) | grep -v 'already satisfied' || true

run: venv
	@$(LOCAL) run app.py

test-simple: venv
	@$(TEST) test/test_healthcheck.py -v

freeze:
	@$(PIP) freeze > requirements.txt

docker-run:
	docker run --name $(APP_NAME) -p 5000:5000 $(APP_NAME)

docker-build:
	docker build --tag $(APP_NAME) .

up: docker-build docker-run
