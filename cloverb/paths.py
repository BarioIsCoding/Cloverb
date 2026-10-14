"""Where external files live; override with environment variables (defaults: the machine this was built on).

CLOVERB_DATA    game stores, benchmark sets, teacher pools   (default /home/user/data)
CLOVERB_MODELS  released Maia-3 checkpoints (maia3/*.pt)     (default /home/user/models)
CLOVERB_EXT     third-party code checkouts (maia3/)          (default /home/user/ext)
"""
import os

DATA = os.environ.get("CLOVERB_DATA", "/home/user/data")
MODELS = os.environ.get("CLOVERB_MODELS", "/home/user/models")
EXT = os.environ.get("CLOVERB_EXT", "/home/user/ext")
