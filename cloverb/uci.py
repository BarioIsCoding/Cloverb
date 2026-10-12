"""Cloverb UCI engine: plays the move a human of a given Elo would play under the given clock.

Inputs the model sees: the player's Elo (option `Elo`), the time control (`BaseTime` seconds, or
auto-detected from the first `go` of a game; increment from `winc`/`binc`), the player's time left
(`wtime`/`btime`), the position and up to 64 previous plies from `position ... moves ...`.

usage: python -m cloverb.uci --model runs/NAME/ema.pt
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np

from .infer import Predictor, bucket_seconds, load

NAME = "Cloverb"


class Engine:
    def __init__(self, a):
        self.a = a
        self.pred = None
        self.elo, self.temp, self.multipv, self.human_time = a.elo, a.temperature, a.multipv, a.human_time
        self.base_opt = a.base_time
        self.reset()

    def reset(self):
        self.fen, self.moves, self.game_base = None, [], None

    def ensure(self):
        if self.pred is None:
            self.pred = Predictor(load(self.a.model, threads=self.a.threads))

    def out(self, s):
        sys.stdout.write(s + "\n"); sys.stdout.flush()

    def setoption(self, toks):
        try:
            i = toks.index("name"); j = toks.index("value")
            name, val = " ".join(toks[i + 1:j]).lower(), " ".join(toks[j + 1:])
        except ValueError:
            return
        if name == "elo": self.elo = float(val)
        elif name == "temperature": self.temp = float(val)
        elif name == "multipv": self.multipv = max(1, min(20, int(val)))
        elif name == "basetime": self.base_opt = float(val)
        elif name == "humantime": self.human_time = val.lower() == "true"

    def position(self, toks):
        self.fen, self.moves = None, []
        if "fen" in toks:
            i = toks.index("fen")
            end = toks.index("moves") if "moves" in toks else len(toks)
            self.fen = " ".join(toks[i + 1:end])
        if "moves" in toks:
            self.moves = toks[toks.index("moves") + 1:]

    def go(self, toks):
        self.ensure()
        kv = {toks[i]: toks[i + 1] for i in range(len(toks) - 1) if toks[i] in
              ("wtime", "btime", "winc", "binc", "movetime", "movestogo")}
        black = bool(self.fen and self.fen.split()[1] == "b") ^ (len(self.moves) % 2 == 1)
        clock = float(kv.get("btime" if black else "wtime", 180000)) / 1000
        inc = float(kv.get("binc" if black else "winc", 0)) / 1000
        if self.base_opt > 0:
            base = self.base_opt
        else:
            if self.game_base is None:  # first `go` of the game: clocks are (close to) the base time
                self.game_base = max(clock, float(kv.get("wtime" if black else "btime", clock * 1000)) / 1000)
            base = self.game_base
        p = self.pred.predict(self.fen, self.moves, self.elo, base, inc, clock, top=max(self.multipv, 32))
        if p is None:
            self.out("bestmove 0000")
            return
        for k in range(min(self.multipv, len(p.moves))):
            cp = int(round(100 * (p.wdl[0] - p.wdl[2]) * 4))
            wdl = " ".join(str(int(round(1000 * x))) for x in p.wdl)
            self.out(f"info depth 1 multipv {k + 1} score cp {cp} wdl {wdl} nodes 1 "
                     f"pv {p.moves[k]} string policy {p.probs[k]:.3f}")
        if self.temp <= 0:
            mv = p.moves[0]
        else:
            w = p.probs ** (1 / self.temp)
            mv = random.choices(p.moves, weights=w / w.sum())[0]
        if self.human_time:
            think = float(bucket_seconds(np.searchsorted(np.cumsum(p.think_dist), random.random())))
            time.sleep(min(think, max(0.0, clock - 1.0) * 0.5))
        self.out(f"bestmove {mv}")

    def loop(self):
        for line in sys.stdin:
            toks = line.split()
            if not toks:
                continue
            cmd = toks[0]
            if cmd == "uci":
                self.out(f"id name {NAME}\nid author Cloverb authors")
                self.out("option name Elo type spin default 1500 min 400 max 3200")
                self.out("option name BaseTime type spin default 0 min 0 max 10800")
                self.out("option name Temperature type string default 0")
                self.out("option name MultiPV type spin default 1 min 1 max 20")
                self.out("option name HumanTime type check default false")
                self.out("uciok")
            elif cmd == "isready":
                self.ensure(); self.out("readyok")
            elif cmd == "ucinewgame":
                self.reset()
            elif cmd == "setoption":
                self.setoption(toks)
            elif cmd == "position":
                self.position(toks)
            elif cmd == "go":
                self.go(toks)
            elif cmd == "quit":
                break


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cloverb UCI engine")
    ap.add_argument("--model", required=True)
    ap.add_argument("--elo", type=float, default=1500)
    ap.add_argument("--base-time", type=float, default=0, help="seconds; 0 = detect from the first go")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--multipv", type=int, default=1)
    ap.add_argument("--human-time", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    Engine(ap.parse_args(argv)).loop()


if __name__ == "__main__":
    main()
