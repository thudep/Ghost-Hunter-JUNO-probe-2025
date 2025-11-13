BASE_URL := https://ghfile.thudep.com:7200
WGET := wget --no-verbose --progress=bar:force:noscroll

geo.h5:
	$(WGET) $(BASE_URL)/geo/geo.h5

concat.h5:
	$(WGET) $(BASE_URL)/test/concat.h5

# 更改以调整分 bin 大小
R ?= 10
THETA ?= 10
T ?= 10

histogram.h5: geo.h5 data
	python3 histogram.py -g $< --data $(word 2,$^) -o $@ -r $(R) -theta $(THETA) -t $(T)

.PHONY: score
score: concat.h5 histogram.h5
	python3 grade.py --concat $<

seeds := $(shell seq 16001 16001) # 更改以调整范围

.PHONY: data
data: $(seeds:%=data/%.h5)

clean:
	rm -rf data/
	rm -rf *.pdf
	rm -rf *.h5
	rm -rf __pycache__

data/%.h5:
	@mkdir -p $(@D)
	$(WGET) -P $(@D) $(BASE_URL)/data/$*.h5