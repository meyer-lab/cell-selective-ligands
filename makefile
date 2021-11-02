SHELL := /bin/bash

.PHONY: clean test

flist = 1 3 4 5 S1 S2 S3 S4

all: output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -Uqr requirements.txt
	touch venv/bin/activate

output/figure%.svg: venv genFigures.py selecv/figures/figure%.py
	mkdir -p ./manuscript/figures
	. venv/bin/activate && ./genFigures.py $*

output/manuscript.md: venv manuscript/*.md
	. venv/bin/activate && manubot process --content-directory=manuscript --output-directory=output --cache-directory=cache --skip-citations --log-level=INFO
	git remote rm rootstock

output/manuscript.html: venv output/manuscript.md
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml \
		output/manuscript.md

output/manuscript.docx: venv output/manuscript.md
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common.yaml \
		--defaults=./common/templates/manubot/pandoc/docx.yaml \
		output/manuscript.md

test: venv
	. venv/bin/activate; pytest -s -v

clean:
	rm -rf output venv
