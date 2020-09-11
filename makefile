SHELL := /bin/bash

.PHONY: clean test

all: pylint.log coverage.xml output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))

flist = 1 2 3 4 5 6 S1 S2

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -Uqr requirements.txt
	touch venv/bin/activate

output/figure%.svg: venv genFigures.py selecv/figures/figure%.py
	mkdir -p ./manuscript/figures
	. venv/bin/activate && JAX_PLATFORM_NAME=cpu ./genFigures.py $*

output/manuscript.md: venv manuscript/*.md
	. venv/bin/activate && manubot process --content-directory=manuscript --output-directory=output --cache-directory=cache --skip-citations --log-level=INFO
	git remote rm rootstock

output/manuscript.html: venv output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml \
		output/manuscript.md

output/manuscript.docx: venv output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/docx.yaml \
		output/manuscript.md

test: venv
	. venv/bin/activate; JAX_PLATFORM_NAME=cpu pytest -s -v

coverage.xml: venv
	. venv/bin/activate; JAX_PLATFORM_NAME=cpu pytest --junitxml=junit.xml --cov=selecv --cov-report xml:coverage.xml

pylint.log: venv
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc selecv > pylint.log || echo "pylint exited with $?")

clean:
	mv output/requests-cache.sqlite requests-cache.sqlite || true
	rm -rf output pylint.log
	mkdir output
	mv requests-cache.sqlite output/requests-cache.sqlite || true
