# Minimal makefile for Sphinx documentation
#

export SPHINX_APIDOC_OPTIONS=members,no-undoc-members,no-show-inheritance

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


buildapi:

	rm -Rf source/_modules/*
	sphinx-apidoc -fMeET -o source/_modules --implicit-namespaces ../gcpds
	#../openbci/acquisition/* ../openbci/database/* ../openbci/preprocess/* ../openbci/stream/*

	mkdir -p source/_notebooks/images
	rm -rf source/_notebooks/*
	mkdir -p source/_notebooks/images

	cp ../notebooks/*.ipynb source/_notebooks/
	#cp ../notebooks/images/* source/_notebooks/images

	#rm source/_notebooks/*.rst
	#rm source/_notebooks/SANDBOX.ipynb
	jupyter nbconvert --to rst source/_notebooks/*.ipynb
	#jupyter nbconvert --to markdown source/_notebooks/readme.ipynb
	#cp source/_notebooks/readme.rst ../README.rst

	jupyter nbconvert --to markdown source/_notebooks/readme.ipynb
	mv source/_notebooks/readme.md ../README.md

	rm -f source/_notebooks/*.ipynb

