#!/usr/bin/env python3
"""
self_improving_learner.py -- a self-improving learner (functional prototype).

Adapted for the nexus_agi repo (emergence/) from Doug's standalone
self-improving-agent (2026-11-10). Stdlib only; self-contained inside
this repo: no imports from other repo modules, no absolute paths, and
no external API calls. The original's three-minds critique step is
replaced here by an optional, clearly-labelled stub
(run_external_critique): it does nothing unless the caller passes a
critique_hook callable, and this module never calls an external API
on its own.

What it does
------------
Evolves a population of small strategy genomes (which learner to use,
polynomial degree, k-NN k, ridge strength, parity subset size) across
four tiny synthetic worlds with hidden rules (a curve, a 3-bit parity,
per-symbol arithmetic, a linear recurrence). Fitness is LEARNING:
generalisation on unseen trials MINUS the ability to fit shuffled
labels (the random-label control). A lineage is kept only if it beats
its parent on a FRESH validation batch each generation, otherwise the
archive rolls back to the parent. Every generation and every written
self-critique is appended to an append-only JSONL memory log. A sealed
test set, generated once from a reserved seed, is touched exactly once
at the end.

Where the ideas came from (read-only research, 2026-11-10)
------------------------------------------------------------
* jennyzzt/dgm (Darwin Goedel Machine): lineage archive +
  accept-only-if-held-out-improves gate.
* noahshinn/reflexion: written self-critique after each generation,
  used only as a one-generation mutation bias (reset every round, so
  the agent cannot learn to game its own self-assessment).
* letta-ai/letta: memory as a first-class persistent part of the agent.
* cbfinn/maml: fitness is ADAPTABILITY to unseen tasks, not score on
  tasks already seen.
* facebookresearch/HyperAgents: the agent's own strategy is the object
  it improves.
* Three disjoint data tiers (train / fresh validation per generation /
  sealed final test) were the fix for selection leakage: one fixed
  holdout would slowly be fitted by selection.

HONEST LIMITS -- read this before believing anything
-----------------------------------------------------
* This is a FUNCTIONAL PROTOTYPE, not true AGI and not consciousness or
  sentience. Nothing here experiences anything.
* The worlds are tiny synthetic tasks. Success here says nothing about
  open-ended real-world intelligence.
* The "self-critique" is rule-based text generated from measured
  diagnostics (scores, weakest world, winning learner). It is not
  introspection and not understanding.
* Evolution tunes strategy choice and hyperparameters; it does not invent
  new algorithms or rewrite its own source code. Self-modifying code is
  deliberately out of scope.
* What is still missing: open-ended tasks, real-world grounding,
  continual learning at scale, common sense, embodiment, values of its
  own, and any agreed test for consciousness (which does not exist).

Memory
------
State lives in `emergence/agent-data/` (a data dir relative to this
repo) by default, or in memory only when AgentMemory(path=None) is
used. Nothing is written outside the repo.

Usage (from the repo root)
-------------------------
  python3 emergence/self_improving_learner.py selftest
  python3 emergence/self_improving_learner.py demo
  python3 emergence/self_improving_learner.py evolve [--generations N]
      [--population N] [--seed S] [--critique]
  python3 emergence/self_improving_learner.py status | recall --key K
      | search --key K | last | surprise | critique

Requires: Python 3.10+, stdlib only. No API keys in this file.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "agent-data"
MEMORY_PATH = DATA_DIR / "memory.jsonl"

# Seeds that have been used for training / validation. The sealed-test seed
# must never appear here until the final evaluation (Claude's fix).
_TOUCHED_SEEDS = set()
SEALED_SEED = 999983  # prime; reserved; touched exactly once, at the end.


def _register_seed(seed: int) -> None:
    _TOUCHED_SEEDS.add(seed)


# ---------------------------------------------------------------------------
# Small linear algebra (adapted from learning-vs-memorization.py)
# ---------------------------------------------------------------------------

def solve_linear(A, b):
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            raise ValueError("singular matrix")
        M[col], M[piv] = M[piv], M[col]
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def ridge_fit(X, y, lam=1e-4):
    """Least-squares fit with ridge regularisation. X rows are features."""
    d = len(X[0])
    XtX = [[0.0] * d for _ in range(d)]
    Xty = [0.0] * d
    for row, yi in zip(X, y):
        for i in range(d):
            Xty[i] += row[i] * yi
            for j in range(d):
                XtX[i][j] += row[i] * row[j]
    for i in range(d):
        XtX[i][i] += lam
    return solve_linear(XtX, Xty)


def predict_linear(beta, x):
    return sum(b * v for b, v in zip(beta, x))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def variance(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def mse(pred, true):
    return sum((p - t) ** 2 for p, t in zip(pred, true)) / max(1, len(true))


def regression_score(pred, true):
    """1 - MSE/var, clipped to [0, 1]. 0 ~ predicting the mean."""
    v = variance(true)
    if v < 1e-12:
        return 1.0 if mse(pred, true) < 1e-12 else 0.0
    return max(0.0, min(1.0, 1.0 - mse(pred, true) / v))


def accuracy(pred, true):
    return sum(1 for p, t in zip(pred, true) if p == t) / max(1, len(true))


# ---------------------------------------------------------------------------
# Worlds -- tiny synthetic tasks with a hidden rule
# Each world: sample(rng, n) -> (X, y); kind in {"regression", "classification"}
# and fit_rule(X, y, genome) -> predictor callable
# ---------------------------------------------------------------------------

class CurveWorld:
    """Hidden curve: y = a*sin(b*x) + c*x^2 + noise. Feature: [x]."""
    name = "curve"
    kind = "regression"

    def __init__(self, rng):
        self.a = rng.uniform(0.3, 0.8)
        self.b = rng.uniform(1.0, 3.0)
        self.c = rng.uniform(0.1, 0.5)

    def hidden(self, x):
        return self.a * math.sin(self.b * x) + self.c * x * x

    def sample(self, rng, n):
        X, y = [], []
        for _ in range(n):
            x = rng.uniform(-2.0, 2.0)
            X.append([x])
            y.append(self.hidden(x) + rng.gauss(0, 0.05))
        return X, y

    def fit_rule(self, X, y, genome):
        deg = int(genome["degree"])
        feats = [[1.0] + [row[0] ** k for k in range(1, deg + 1)] for row in X]
        beta = ridge_fit(feats, y, lam=genome["ridge"])
        return lambda row: predict_linear(
            beta, [1.0] + [row[0] ** k for k in range(1, deg + 1)])


class ParityWorld:
    """Hidden parity of a 3-bit subset of 10 bits. Feature: the 10 bits.
    10 bits (1,024 patterns) so a memorizer cannot cover the space."""
    name = "parity"
    kind = "classification"

    def __init__(self, rng):
        self.subset = tuple(sorted(rng.sample(range(10), 3)))

    def hidden(self, bits):
        return sum(bits[i] for i in self.subset) % 2

    def sample(self, rng, n):
        X, y = [], []
        for _ in range(n):
            bits = [rng.randint(0, 1) for _ in range(10)]
            X.append([float(b) for b in bits])
            y.append(self.hidden(bits))
        return X, y

    def fit_rule(self, X, y, genome):
        # Occam penalty: a large hypothesis library can partly fit ANY
        # labels by chance (multiple comparisons), so a bigger subset must
        # earn its keep: each extra bit costs 5 points of accuracy.
        # Score = accuracy - 0.05 * subset size.
        max_size = int(genome["subset"])
        best, best_pen = None, -1.0
        from itertools import combinations
        for size in range(1, max_size + 1):
            for combo in combinations(range(10), size):
                acc = accuracy(
                    [sum(int(row[i]) for i in combo) % 2 for row in X], y)
                pen = acc - 0.05 * size
                if pen > best_pen:
                    best, best_pen = combo, pen
        combo = best or (0,)
        return lambda row: sum(int(row[i]) for i in combo) % 2


class CompositionWorld:
    """Hidden arithmetic op per symbol: A/B/C map to + / - / *.
    Feature: [a, b, symbol_index]. Regression on the result."""
    name = "composition"
    kind = "regression"
    OPS = {"add": lambda a, b: a + b,
           "sub": lambda a, b: a - b,
           "mul": lambda a, b: a * b}

    def __init__(self, rng):
        ops = rng.sample(list(self.OPS), 3)
        self.mapping = {0: ops[0], 1: ops[1], 2: ops[2]}

    def hidden(self, a, b, s):
        return float(self.OPS[self.mapping[s]](a, b))

    def sample(self, rng, n):
        X, y = [], []
        for _ in range(n):
            a, b = rng.randint(0, 9), rng.randint(0, 9)
            s = rng.randint(0, 2)
            X.append([float(a), float(b), float(s)])
            y.append(self.hidden(a, b, s) + rng.gauss(0, 0.05))
        return X, y

    def fit_rule(self, X, y, genome):
        per_symbol = {}
        for s in range(3):
            rows = [(row, t) for row, t in zip(X, y) if int(row[2]) == s]
            if not rows:
                per_symbol[s] = "add"
                continue
            best_op, best_err = "add", float("inf")
            for name, fn in self.OPS.items():
                err = mse([float(fn(int(r[0]), int(r[1]))) for r, _ in rows],
                          [t for _, t in rows])
                if err < best_err:
                    best_op, best_err = name, err
            per_symbol[s] = best_op
        return lambda row: float(
            self.OPS[per_symbol[int(row[2])]](int(row[0]), int(row[1])))


class SequenceWorld:
    """Hidden linear recurrence: s_t = p*s_{t-1} + q*s_{t-2} + noise.
    Feature: [s_{t-1}, s_{t-2}]. Regression on s_t."""
    name = "sequence"
    kind = "regression"

    def __init__(self, rng):
        self.p = rng.uniform(-0.9, 0.9)
        self.q = rng.uniform(-0.9, 0.9)

    def sample(self, rng, n):
        X, y = [], []
        s_prev, s_prev2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        for _ in range(n):
            s = self.p * s_prev + self.q * s_prev2 + rng.gauss(0, 0.02)
            X.append([s_prev, s_prev2])
            y.append(s)
            s_prev2, s_prev = s_prev, s
        return X, y

    def fit_rule(self, X, y, genome):
        feats = [[1.0, row[0], row[1]] for row in X]
        beta = ridge_fit(feats, y, lam=genome["ridge"])
        return lambda row: predict_linear(beta, [1.0, row[0], row[1]])


WORLDS = [CurveWorld, ParityWorld, CompositionWorld, SequenceWorld]


def make_worlds(seed):
    """Fresh hidden rules, drawn once per run. Deterministic in `seed`."""
    rng = random.Random(seed)
    return [cls(rng) for cls in WORLDS]


def draw_split(world, seed, n_train=60, n_val=40):
    """Train split + one fresh validation batch from a NEW seed."""
    _register_seed(seed)
    rng = random.Random(seed)
    return world.sample(rng, n_train), world.sample(rng, n_val)

# ---------------------------------------------------------------------------
# Learners -- the three strategies from the learning lab
# ---------------------------------------------------------------------------

def _key(row):
    return tuple(round(float(v), 6) for v in row)


class Memorizer:
    """Stores training pairs; exact match or fallback. The control case."""

    def __init__(self):
        self.table, self.fallback = {}, None

    def fit(self, X, y, kind, genome, world=None):
        self.table = {_key(r): t for r, t in zip(X, y)}
        self.fallback = (mean(y) if kind == "regression"
                         else max(set(y), key=list(y).count))

    def predict(self, X):
        return [self.table.get(_key(r), self.fallback) for r in X]


class NearestNeighbour:
    """k-NN over feature vectors. Local similarity, no rule."""

    def __init__(self):
        self.X = self.y = None

    def fit(self, X, y, kind, genome, world=None):
        self.X, self.y = X, y

    def _nearest(self, row, k):
        d = sorted(range(len(self.X)),
                   key=lambda i: sum((a - b) ** 2
                                     for a, b in zip(self.X[i], row)))
        return [self.y[i] for i in d[:k]]

    def predict(self, X, kind="regression", genome=None):
        k = int(genome["k"]) if genome else 3
        out = []
        for row in X:
            ns = self._nearest(row, k)
            out.append(mean(ns) if kind == "regression"
                       else max(set(ns), key=ns.count))
        return out


class RuleInducer:
    """Asks the world for its hypothesis class and fits it. The true learner."""

    def __init__(self):
        self.fn = None

    def fit(self, X, y, kind, genome, world=None):
        self.fn = world.fit_rule(X, y, genome)

    def predict(self, X):
        return [self.fn(r) for r in X]


LEARNERS = {"memorizer": Memorizer, "nearest": NearestNeighbour,
            "rule": RuleInducer}


def learner_for(genome):
    cls = LEARNERS[genome["learner"]]
    return cls()


def score_learner(learner, X, y, kind, genome):
    pred = (learner.predict(X, kind, genome)
            if isinstance(learner, NearestNeighbour)
            else learner.predict(X))
    if kind == "regression":
        return regression_score(pred, y)
    return accuracy([int(round(p)) for p in pred], [int(t) for t in y])


def fit_random_control(learner_cls, world, kind, genome, seed, n=150):
    """Fit the SAME learner to SHUFFLED labels and measure how well it fits
    them. A memorizer fits anything (score ~1); a rule learner cannot.
    This is the learning lab's random-label control, adapted.
    The control draws its own fresh sample (n=150): on the small 60-sample
    train set, duplicate input patterns with conflicting shuffled labels
    would cap the memorizer's score for a bookkeeping reason, not a
    learning reason -- the control must measure capacity, not collisions."""
    rng = random.Random(seed)
    Xc, yc = world.sample(rng, n)
    ys = yc[:]
    rng.shuffle(ys)
    learner = learner_cls()
    learner.fit(Xc, ys, kind, genome, world)
    return score_learner(learner, Xc, ys, kind, genome)


def evaluate_genome(genome, worlds, train_sets, val_seed):
    """Return per-world dicts: {gen, random_fit, learning}.
    Validation is FRESH (new seed each call) -- Claude's fix."""
    per_world = {}
    for world, (Xtr, ytr) in zip(worlds, train_sets):
        _register_seed(val_seed)
        # Stable per-world offset (never hash(), which is salted per run).
        off = sum(ord(c) for c in world.name)
        vrng = random.Random(val_seed + off)
        Xv, yv = world.sample(vrng, 40)
        learner = learner_for(genome)
        learner.fit(Xtr, ytr, world.kind, genome, world)
        gen = score_learner(learner, Xv, yv, world.kind, genome)
        rfit = fit_random_control(LEARNERS[genome["learner"]], world,
                                  world.kind, genome, val_seed + 7)
        per_world[world.name] = {
            "gen": round(gen, 4),
            "random_fit": round(rfit, 4),
            "learning": round(max(-1.0, min(1.0, gen - rfit)), 4),
        }
    fit = mean([v["learning"] for v in per_world.values()])
    return {"fitness": round(fit, 4),
            "mean_gen": round(mean([v["gen"] for v in per_world.values()]), 4),
            "mean_random_fit": round(
                mean([v["random_fit"] for v in per_world.values()]), 4),
            "per_world": per_world}


# ---------------------------------------------------------------------------
# Evolution -- genome, mutation, lineage archive (from meta-algorithm-generator
# and the Darwin Goedel Machine idea; accept-only-if-better gate)
# ---------------------------------------------------------------------------

def random_genome(rng):
    return {"learner": rng.choice(list(LEARNERS)),
            "degree": rng.randint(2, 4),
            "k": rng.randint(1, 5),
            "ridge": 10 ** rng.uniform(-4, -1),
            "subset": rng.randint(1, 3)}


def mutate(genome, rng, bias=None):
    """One-field mutation. `bias` is a list of (field, direction) directives
    from the latest self-critique; it is consumed here and RESET by the
    caller after each generation (Groq's fix)."""
    g = dict(genome)
    if bias and rng.random() < 0.6:
        field, direction = rng.choice(bias)
        if field == "learner" and isinstance(direction, str):
            g["learner"] = direction
        elif field in ("degree", "k", "subset"):
            lo, hi = {"degree": (1, 5), "k": (1, 7), "subset": (1, 3)}[field]
            g[field] = max(lo, min(hi, g[field] + direction))
        elif field == "ridge":
            g["ridge"] = max(1e-5, min(1.0, g["ridge"] * (10 ** direction)))
        return g
    field = rng.choice(["learner", "degree", "k", "ridge", "subset"])
    if field == "learner":
        g["learner"] = rng.choice(list(LEARNERS))
    elif field == "degree":
        g["degree"] = max(1, min(5, g["degree"] + rng.choice([-1, 1])))
    elif field == "k":
        g["k"] = max(1, min(7, g["k"] + rng.choice([-1, 1])))
    elif field == "ridge":
        g["ridge"] = max(1e-5, min(1.0, g["ridge"] * (10 ** rng.choice([-1, 1]))))
    else:
        g["subset"] = max(1, min(3, g["subset"] + rng.choice([-1, 1])))
    return g


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def tournament(pop, scored, rng, size=3):
    cand = rng.sample(range(len(pop)), min(size, len(pop)))
    return pop[max(cand, key=lambda i: scored[i]["fitness"])]


# ---------------------------------------------------------------------------
# Reflection -- Reflexion-style written self-critique (rule-based text from
# measured diagnostics; NOT introspection)
# ---------------------------------------------------------------------------

def reflect(generation, best, prev_best, per_world, rng):
    """Write a short self-critique and a mutation-bias buffer."""
    weak = min(per_world, key=lambda w: per_world[w]["learning"])
    strong = max(per_world, key=lambda w: per_world[w]["learning"])
    w, s = per_world[weak], per_world[strong]
    lines = [f"Generation {generation} self-critique."]
    bias = []
    if best["mean_random_fit"] > 0.6:
        lines.append(
            f"Warning: mean random-label fit is {best['mean_random_fit']:.2f} "
            "-- the winning strategy can fit shuffled labels, which smells "
            "of memorisation, not learning.")
        bias.append(("learner", "rule"))
    if w["learning"] < 0.3:
        lines.append(
            f"Weakest world is {weak} (learning {w['learning']:.2f}, "
            f"generalisation {w['gen']:.2f}, random fit "
            f"{w['random_fit']:.2f}).")
        if weak == "curve":
            lines.append("The curve needs a richer hypothesis -- try a "
                         "higher polynomial degree.")
            bias.append(("degree", 1))
        elif weak == "parity":
            lines.append("Parity needs wider subset search -- try larger "
                         "subsets.")
            bias.append(("subset", 1))
        elif weak == "sequence":
            lines.append("The recurrence fit is stiff -- try less ridge "
                         "regularisation.")
            bias.append(("ridge", -1))
        else:
            lines.append("Composition confuses symbols -- try nearest "
                         "neighbours for local structure.")
            bias.append(("learner", "nearest"))
    lines.append(f"Strongest world is {strong} (learning "
                 f"{s['learning']:.2f}). Keep what works there.")
    if prev_best is not None and best["fitness"] <= prev_best["fitness"]:
        lines.append("No improvement over the parent lineage this round; "
                     "the archive keeps the parent (rollback).")
    if best["fitness"] > 0.6:
        lines.append("Learning score is genuinely high across worlds -- "
                     "this lineage is behaving like a true learner.")
    return " ".join(lines), bias


# ---------------------------------------------------------------------------
# Memory -- append-only JSONL log + inverted index (from CKB)
# ---------------------------------------------------------------------------

def _tokens(text):
    return [t for t in "".join(
        ch.lower() if ch.isalnum() else " " for ch in text).split()
        if len(t) > 2]


class AgentMemory:
    """Append-only log of generations, reflections and critiques."""

    def __init__(self, path=MEMORY_PATH):
        # path=None means in-memory only: nothing is written to disk.
        # The default path is a data dir relative to this repo
        # (emergence/agent-data/).
        self.path = Path(path) if path is not None else None
        self.records = []
        self.index = {}
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._ingest(json.loads(line), persist=False)

    def _ingest(self, rec, persist=True):
        rec.setdefault("id", len(self.records) + 1)
        self.records.append(rec)
        for tok in set(_tokens(json.dumps(rec))):
            self.index.setdefault(tok, []).append(rec["id"])
        if persist and self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def log(self, kind, **fields):
        rec = {"kind": kind, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime()),
               **fields}
        self._ingest(rec)
        return rec

    def search(self, key):
        ids = set()
        for tok in _tokens(key):
            ids.update(self.index.get(tok, []))
        return [self.records[i - 1] for i in sorted(ids)]

    def last(self, key=None):
        if key is None:
            return self.records[-1] if self.records else None
        hits = self.search(key)
        return hits[-1] if hits else None

    def stats(self):
        kinds = {}
        for r in self.records:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        return {"records": len(self.records), "kinds": kinds,
                "index_terms": len(self.index)}

    def surprise(self, rng):
        return rng.choice(self.records) if self.records else None

# ---------------------------------------------------------------------------
# The agent -- evolve loop with lineage archive + sealed final test
# ---------------------------------------------------------------------------

def sealed_test(genome, worlds, seed=SEALED_SEED):
    """Evaluate on the sealed set. MUST be called exactly once per run,
    after all evolution is done. Asserts the seed was never touched."""
    assert seed not in _TOUCHED_SEEDS, (
        "sealed seed was already used -- the final test is compromised")
    _register_seed(seed)
    return _sealed_eval(genome, worlds, seed)


def _sealed_eval(genome, worlds, seed):
    rng = random.Random(seed)
    per_world = {}
    for world in worlds:
        Xtr, ytr = world.sample(rng, 60)
        Xte, yte = world.sample(rng, 100)
        learner = learner_for(genome)
        learner.fit(Xtr, ytr, world.kind, genome, world)
        gen = score_learner(learner, Xte, yte, world.kind, genome)
        rfit = fit_random_control(LEARNERS[genome["learner"]], world,
                                  world.kind, genome, seed + 13)
        per_world[world.name] = {
            "gen": round(gen, 4), "random_fit": round(rfit, 4),
            "learning": round(max(-1.0, min(1.0, gen - rfit)), 4)}
    return {"fitness": round(mean([v["learning"] for v in per_world.values()]), 4),
            "mean_gen": round(mean([v["gen"] for v in per_world.values()]), 4),
            "mean_random_fit": round(
                mean([v["random_fit"] for v in per_world.values()]), 4),
            "per_world": per_world}


def evolve(seed=42, generations=5, population=8, mem=None, verbose=True,
           run_critique=False):
    """Run the evolutionary loop. Returns (best_genome, report)."""
    rng = random.Random(seed)
    worlds = make_worlds(seed)
    train_sets = []
    for i, world in enumerate(worlds):
        tr_seed = seed * 1000 + i
        _register_seed(tr_seed)
        trng = random.Random(tr_seed)
        train_sets.append(world.sample(trng, 60))
    mem = mem or AgentMemory()
    pop = [random_genome(rng) for _ in range(population)]
    lineage_best, lineage_scored = None, None
    history = []

    for gen in range(generations):
        val_seed = seed * 100000 + gen * 7919 + 104729  # fresh each gen
        scored = [evaluate_genome(g, worlds, train_sets, val_seed)
                  for g in pop]
        best_i = max(range(len(pop)), key=lambda i: scored[i]["fitness"])
        best, best_scored = pop[best_i], scored[best_i]
        # Lineage archive gate: accept only if it beats the parent lineage
        # on THIS generation's fresh validation (DGM-style).
        prev_lineage = lineage_scored
        accepted = (lineage_scored is None
                    or best_scored["fitness"] > lineage_scored["fitness"])
        if accepted:
            lineage_best, lineage_scored = dict(best), best_scored
        critique_text, bias = reflect(gen, best_scored,
                                      prev_lineage,
                                      best_scored["per_world"], rng)
        mem.log("generation", generation=gen, accepted=accepted,
                genome=best, scores=best_scored)
        mem.log("reflection", generation=gen, text=critique_text,
                bias=bias)
        # History tracks the SURVIVING lineage, so it never regresses.
        history.append({"generation": gen, "accepted": accepted,
                        "fitness": lineage_scored["fitness"],
                        "gen": lineage_scored["mean_gen"],
                        "random_fit": lineage_scored["mean_random_fit"],
                        "genome": dict(lineage_best)})
        if verbose:
            print(f"gen {gen}: fitness {best_scored['fitness']:+.3f} "
                  f"(gen {best_scored['mean_gen']:.3f}, "
                  f"random-fit {best_scored['mean_random_fit']:.3f}) "
                  f"learner={best['learner']} "
                  f"{'ACCEPTED' if accepted else 'rollback->parent'}")
            print(f"    critique: {critique_text}")
        # Next generation: elite parent + children. Bias buffer is used
        # here and then RESET (Groq's fix) -- it never survives a generation.
        children = [dict(lineage_best)]
        while len(children) < population:
            parent = tournament(pop, scored, rng)
            other = tournament(pop, scored, rng)
            child = crossover(parent, other, rng)
            child = mutate(child, rng, bias=bias)
            children.append(child)
        pop = children
        bias = []  # reset: the critique never steers more than one round
    if run_critique:
        run_external_critique(mem, lineage_best, lineage_scored)
    return lineage_best, {"history": history,
                          "best_scores": lineage_scored,
                          "memory": mem.stats()}


def run_external_critique(mem, genome=None, scores=None, critique_hook=None):
    """Optional external review of the latest lineage.

    STUB -- does nothing on its own. This module never calls an external
    API. If the caller supplies a `critique_hook` callable, it is given
    one text snapshot (genome + per-world scores + random-label control)
    and whatever text it returns is logged to memory. With no hook
    configured, this prints a short note and returns None.
    """
    if critique_hook is None:
        print("critique: no external critique hook configured; skipping "
              "(this module never calls an external API on its own)")
        return None
    if genome is None or scores is None:
        rec = None
        for r in reversed(mem.records):
            if r.get("kind") == "generation":
                rec = r
                break
        if rec is None:
            print("critique: no generation in memory yet; run evolve first")
            return None
        genome, scores = rec.get("genome"), rec.get("scores")
        gen_no = rec.get("generation")
    else:
        gen_no = None
    snap = ("Self-improving-learner status"
            + (f" after generation {gen_no}" if gen_no is not None else "")
            + f". Best genome: {json.dumps(genome)}. "
              f"Scores (fitness = generalisation minus random-label fit, "
              f"fresh validation each generation): {json.dumps(scores)}. "
              "NOTE on how the genome is used: each world has its OWN "
              "hypothesis class -- the polynomial degree applies to the "
              "curve world only; parity is learned by XOR-subset search, "
              "composition by per-symbol op selection, sequence by a "
              "linear-recurrence fit. The genome fields are shared, but "
              "each world uses only its own. "
              "In under 120 words: is this lineage a true learner or a "
              "memoriser, and what single change would you make next?")
    try:
        reply = critique_hook(snap)
    except Exception as e:
        print(f"critique: hook failed ({e}); skipping")
        return None
    if not reply:
        print("critique: hook returned nothing; skipping")
        return None
    mem.log("critique", text=str(reply)[:4000])
    print("critique: external reply logged to memory.")
    print(str(reply)[:1500])
    return reply


# ---------------------------------------------------------------------------
# Selftest + demo
# ---------------------------------------------------------------------------

def selftest(out=print):
    """Offline checks, no API calls. Returns True if all pass."""
    checks = []

    def check(name, ok):
        checks.append((name, bool(ok)))
        out(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    out("self-improving-learner selftest")
    # 1. Deterministic worlds
    w1 = make_worlds(42)
    w2 = make_worlds(42)
    rng = random.Random(7)
    a = [w.sample(rng, 10) for w in w1]
    rng = random.Random(7)
    b = [w.sample(rng, 10) for w in w2]
    check("worlds are deterministic for a fixed seed", a == b)

    # 2. Rule beats memorizer on unseen parity; memorizer fits random labels
    worlds = make_worlds(123)
    parity = worlds[1]
    rng = random.Random(11)
    Xtr, ytr = parity.sample(rng, 60)
    Xte, yte = parity.sample(rng, 100)
    g_rule = {"learner": "rule", "degree": 2, "k": 3, "ridge": 1e-4,
              "subset": 3}
    g_mem = {"learner": "memorizer", "degree": 2, "k": 3, "ridge": 1e-4,
             "subset": 3}
    r = RuleInducer(); r.fit(Xtr, ytr, "classification", g_rule, parity)
    m = Memorizer(); m.fit(Xtr, ytr, "classification", g_mem, parity)
    check("rule generalises on parity (>0.9)",
          accuracy([int(round(p)) for p in r.predict(Xte)],
                   [int(t) for t in yte]) > 0.9)
    check("memorizer generalises poorly on parity (<0.7)",
          accuracy([int(round(p)) for p in m.predict(Xte)],
                   [int(t) for t in yte]) < 0.7)
    rf_rule = fit_random_control(RuleInducer, parity, "classification",
                                 g_rule, 5)
    rf_mem = fit_random_control(Memorizer, parity, "classification",
                                g_mem, 5)
    check("rule cannot fit shuffled labels (<0.7)", rf_rule < 0.7)
    check("memorizer fits shuffled labels (>0.9)", rf_mem > 0.9)

    # 3. Ridge recovers known coefficients
    rng = random.Random(3)
    X = [[1.0, rng.uniform(-1, 1)] for _ in range(50)]
    y = [2.0 + 3.0 * row[1] for row in X]
    beta = ridge_fit(X, y, lam=1e-6)
    check("ridge recovers y = 2 + 3x",
          abs(beta[0] - 2.0) < 0.05 and abs(beta[1] - 3.0) < 0.05)

    # 4. Curve: rule generalises better than memorizer
    curve = worlds[0]
    rng = random.Random(21)
    Xtr, ytr = curve.sample(rng, 60)
    Xte, yte = curve.sample(rng, 100)
    r = RuleInducer(); r.fit(Xtr, ytr, "regression", g_rule, curve)
    m = Memorizer(); m.fit(Xtr, ytr, "regression", g_mem, curve)
    check("rule generalises better than memorizer on curve",
          regression_score(r.predict(Xte), yte)
          > regression_score(m.predict(Xte), yte))

    # 5. Evolution does not regress (lineage gate) and memory works.
    # In-memory memory: the selftest writes no files into the repo.
    mem = AgentMemory(path=None)
    best, rep = evolve(seed=42, generations=3, population=6, mem=mem,
                       verbose=False)
    hist = rep["history"]
    check("evolution lineage never regresses",
          all(hist[i + 1]["fitness"] >= hist[i]["fitness"] - 1e-9
              for i in range(len(hist) - 1)))
    check("memory recorded generations", mem.stats()["records"] >= 6)
    rec = mem.last("reflection")
    check("recall finds the latest reflection",
          rec is not None and "self-critique" in rec.get("text", ""))
    check("search finds generation records", len(mem.search("generation")) > 0)
    check("best genome is a valid genome",
          best["learner"] in LEARNERS and 1 <= best["degree"] <= 5)

    # 6. Sealed seed untouched until final evaluation
    check("sealed seed untouched before final test",
          SEALED_SEED not in _TOUCHED_SEEDS)
    # 7. Reflection is non-empty and bias buffer resets
    _, bias = reflect(0, {"fitness": 0.1, "mean_gen": 0.4,
                          "mean_random_fit": 0.9,
                          "per_world": {w.name: {"gen": 0.4, "random_fit": 0.9,
                                                 "learning": -0.5}
                                        for w in worlds}},
                      None,
                      {w.name: {"gen": 0.4, "random_fit": 0.9,
                                "learning": -0.5} for w in worlds},
                      random.Random(1))
    check("reflection produces text + bias", bool(bias))

    # 8. This file compiles
    import py_compile
    try:
        py_compile.compile(str(Path(__file__).resolve()), doraise=True)
        check("this file compiles cleanly", True)
    except Exception:
        check("this file compiles cleanly", False)

    n_pass = sum(1 for _, ok in checks if ok)
    out(f"selftest: {n_pass}/{len(checks)} PASS")
    return n_pass == len(checks)


def run_demo():
    """Evolve, then measure honestly on the sealed set. Prints real numbers."""
    print("== self-improving-learner demo ==")
    print("Evolving strategy genomes across 4 hidden-rule worlds; fitness "
          "is LEARNING (generalisation minus random-label fit), measured on "
          "fresh validation every generation. Sealed test touched once, "
          "at the end.")
    mem = AgentMemory()
    # Baseline: a random first genome, measured on the sealed set BEFORE
    # evolution -- but the sealed set may only be touched once, so we
    # instead report the generation-0 fresh-validation numbers as the
    # "before" and the sealed numbers as the "after".
    best, rep = evolve(seed=42, generations=5, population=8, mem=mem,
                       verbose=True)
    worlds = make_worlds(42)
    sealed = _sealed_eval(best, worlds, SEALED_SEED)
    first = rep["history"][0]
    print("\n== results ==")
    print(f"before (gen 0, fresh validation): "
          f"fitness {first['fitness']:+.3f}, generalisation "
          f"{first['gen']:.3f}, random-fit {first['random_fit']:.3f}")
    print(f"after  (best lineage, SEALED test): "
          f"fitness {sealed['fitness']:+.3f}, generalisation "
          f"{sealed['mean_gen']:.3f}, random-fit {sealed['mean_random_fit']:.3f}")
    print(f"best genome: {best}")
    print("per-world sealed scores (gen / random-fit / learning):")
    for name, v in sealed["per_world"].items():
        print(f"  {name:12s} {v['gen']:.3f} / {v['random_fit']:.3f} / "
              f"{v['learning']:+.3f}")
    verdict = ("TRUE LEARNER" if sealed["mean_gen"] >= 0.75
               and sealed["mean_random_fit"] <= 0.5 else
               "MEMORIZER" if sealed["mean_random_fit"] >= 0.8
               else "DEVELOPING")
    print(f"verdict on sealed test: {verdict}")
    print(f"memory: {mem.stats()}")
    print("\nHonest limits: functional prototype on 4 tiny synthetic worlds; "
          "not AGI, not consciousness. The sealed test is one small sample, "
          "not proof of general intelligence.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="self-improving-learner: evolve strategies, check for "
                    "real learning, remember everything.")
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest", help="offline checks, no API calls")
    sp.add_parser("demo", help="evolve + sealed-test demo with real numbers")
    e = sp.add_parser("evolve", help="run the evolutionary loop")
    e.add_argument("--generations", type=int, default=5)
    e.add_argument("--population", type=int, default=8)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--critique", action="store_true",
                   help="ask the configured external critique hook at the end "
                   "(none configured by default; does nothing)")
    sp.add_parser("status", help="memory stats")
    r = sp.add_parser("recall", help="newest record mentioning a key")
    r.add_argument("--key", required=True)
    s = sp.add_parser("search", help="records mentioning a key")
    s.add_argument("--key", required=True)
    sp.add_parser("last", help="newest memory record")
    sp.add_parser("surprise", help="a random memory record")
    sp.add_parser("critique", help="ask the configured external critique "
                                  "hook (stub: does nothing unless a hook "
                                  "is configured)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "selftest":
        return 0 if selftest() else 1
    if args.cmd == "demo":
        run_demo()
        return 0
    if args.cmd == "evolve":
        mem = AgentMemory()
        best, rep = evolve(seed=args.seed, generations=args.generations,
                           population=args.population, mem=mem,
                           run_critique=args.critique)
        print(f"best genome: {best}")
        print(f"best scores: {rep['best_scores']}")
        return 0
    mem = AgentMemory()
    if args.cmd == "status":
        print(json.dumps(mem.stats(), indent=2))
    elif args.cmd == "recall":
        rec = mem.last(args.key)
        print(json.dumps(rec, indent=2) if rec else "no record found")
    elif args.cmd == "search":
        hits = mem.search(args.key)
        for h in hits:
            print(json.dumps(h))
        print(f"{len(hits)} record(s)")
    elif args.cmd == "last":
        rec = mem.last()
        print(json.dumps(rec, indent=2) if rec else "memory is empty")
    elif args.cmd == "surprise":
        rec = mem.surprise(random.Random())
        print(json.dumps(rec, indent=2) if rec else "memory is empty")
    elif args.cmd == "critique":
        run_external_critique(mem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
