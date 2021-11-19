SHELL := /bin/bash

.PHONY: clean test

flist = 1 2 3 4 5 S1 S2 S3 O1

all: $(patsubst %, output/figure%.svg, $(flist))

output/figure%.svg: selecv/figures/figure%.py
	@mkdir -p output
	poetry run fbuild $*

test:
	poetry run pytest -s -v -x

clean:
	rm -rf output
