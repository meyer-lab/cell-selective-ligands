SHELL := /bin/bash

.PHONY: clean test

flist = 1 2 3 4 5 6 7 S1 S2

all: output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))

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
		--csl=./manuscript/integrative-biology.csl \
		output/manuscript.md

output/manuscript.docx: venv output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/docx.yaml \
		--csl=./manuscript/integrative-biology.csl \
		output/manuscript.md

test: venv
	. venv/bin/activate; JAX_PLATFORM_NAME=cpu pytest -s -v

clean:
	mv output/requests-cache.sqlite requests-cache.sqlite || true
	rm -rf output
	mkdir output
	mv requests-cache.sqlite output/requests-cache.sqlite || true
