# Specify source tex files here
TEX_SRCS=examples/beamerexample-conference-talk.tex

# The following variables may be overridden on the make command line in
PERL := perl
LATEXMK := $(PERL) scripts/latexmk.pl
BUILD_DIR := build
PDF_DIR := pdf

LATEXMK_CMD=TEXINPUTS="$$TEXINPUTS:cambridge-beamer" $(LATEXMK) -pdf -pdflatex="xelatex %O %S"

.DEFAULT: all
.PHONY: all
all: $(addprefix $(PDF_DIR)/, $(TEX_SRCS:.tex=.pdf))

.PHONY: clean
clean:
	rm -rf "$(BUILD_DIR)" "$(PDF_DIR)"

$(PDF_DIR)/%.pdf: %.tex
	mkdir -p "$(BUILD_DIR)/$*"
	$(LATEXMK_CMD) -outdir="$(BUILD_DIR)/$*" "$<"
	mkdir -p "$(dir $@)"
	cp "$(BUILD_DIR)/$*/$(notdir $*).pdf" "$@"
