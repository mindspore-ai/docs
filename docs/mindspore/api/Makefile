# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source_zh_cn
BUILDDIR      = build_zh_cn

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

EXTRADIR      = $(SOURCEDIR)/api_python

.PHONY: clean

clean:
	-rm -rf $(BUILDDIR)/* $(EXTRADIR)/ops $(EXTRADIR)/nn $(EXTRADIR)/dataset $(EXTRADIR)/dataset_vision $(EXTRADIR)/dataset_audio $(EXTRADIR)/dataset_transforms $(EXTRADIR)/dataset_text $(EXTRADIR)/text $(EXTRADIR)/numpy $(EXTRADIR)/nn_probability $(EXTRADIR)/mindspore $(EXTRADIR)/scipy
