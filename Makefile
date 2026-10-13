# Native builds. `make` builds the standalone engine and the shared libraries used by the Python code.
CC ?= gcc
CFLAGS ?= -O3 -march=native

all: cloverb-engine cloverb/libcloverb.so cloverb/libattn64.so

# Standalone UCI engine (no Python / PyTorch). Needs AVX-512; uses AMX bf16 tiles when available.
cloverb-engine: csrc/cloverb_engine.c csrc/cloverb_core.c
	$(CC) $(CFLAGS) -fopenmp -o $@ csrc/cloverb_engine.c -lm

cloverb/libcloverb.so: csrc/cloverb_core.c
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $<

cloverb/libattn64.so: csrc/attn64.c
	$(CC) $(CFLAGS) -fopenmp -shared -fPIC -o $@ $< -lm

# Export trained weights for the standalone engine: make weights CKPT=runs/final/ema.pt
weights:
	python tools/export_engine.py $(CKPT) checkpoints/cloverb.bin

clean:
	rm -f cloverb-engine cloverb/libcloverb.so cloverb/libattn64.so

.PHONY: all weights clean
