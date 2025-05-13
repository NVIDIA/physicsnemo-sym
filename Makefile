install:
	pip install --upgrade pip && \
		pip install -e . --no-build-isolation

setup-ci:
	pip install pre-commit && \
	pre-commit install

black:
	pre-commit run black -a

interrogate:
	# pre-commit run interrogate -a
	echo "Interrogate CI stage not currently implemented"

lint:
	# pre-commit run markdownlint -a
	echo "Lint CI stage not currently implemented"

license:
	pre-commit run license -a

doctest:
	# coverage run \
	# 	--rcfile='test/coverage.docstring.rc' \
	# 	-m pytest \
	# 	--doctest-modules physicsnemo/ --ignore-glob=*internal*
	echo "Doctest CI stage not currently implemented"

pytest:
	coverage run \
		--rcfile='test/coverage.pytest.rc' \
		-m pytest --ignore-glob=*docs* --ignore=physicsnemo/sym/utils/vpinn/test_functions.py --ignore-glob=*examples*

coverage:
	coverage combine && \
		coverage report --show-missing --omit=*test* --omit=*internal* --fail-under=50 && \
		coverage html

all-ci: setup-ci black interrogate lint license install pytest doctest coverage

# ============================================================================ #
# CLEAN COMMANDS
# ============================================================================ #

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage*
	rm -fr htmlcov/
	rm -fr .pytest_cache

# For arch naming conventions, refer
# https://docs.docker.com/build/building/multi-platform/
# https://github.com/containerd/containerd/blob/v1.4.3/platforms/platforms.go#L86
ARCH := $(shell uname -p)

ifeq ($(ARCH), x86_64)
    TARGETPLATFORM := "linux/amd64"
else ifeq ($(ARCH), aarch64)
    TARGETPLATFORM := "linux/arm64"
else
    $(error Unknown CPU architecture ${ARCH} detected)
endif

PHYSICSNEMO_SYM_GIT_HASH = $(shell git rev-parse --short HEAD)

container-deploy:
	docker build -t physicsnemo-sym:deploy --build-arg TARGETPLATFORM=${TARGETPLATFORM} --build-arg PHYSICSNEMO_SYM_GIT_HASH=${PHYSICSNEMO_SYM_GIT_HASH} --target deploy -f Dockerfile .

container-ci:
	docker build -t physicsnemo-sym:ci --build-arg TARGETPLATFORM=${TARGETPLATFORM} --target ci -f Dockerfile .

container-docs:
	docker build -t physicsnemo-sym:docs --build-arg TARGETPLATFORM=${TARGETPLATFORM} --target docs -f Dockerfile .
