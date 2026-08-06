#!/usr/bin/env python3

import numpy as np

SWC = "/Users/jhsinger/Documents/nNOS_AC_2026/EyewireAnalysis/720575940584776059/skeleton.swc"

rows = []

with open(SWC) as f:
    for line in f:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        p = line.split()
        if len(p) < 7:
            continue

        rows.append((
            int(p[0]),
            int(p[6])
        ))

rows = np.array(rows)

ids = rows[:,0].astype(int)
parents = rows[:,1].astype(int)

N = len(ids)

roots = np.sum(parents == -1)

edges = np.sum(parents != -1)

print()
print("Nodes:", N)
print("Edges:", edges)
print("Roots:", roots)

print()

if roots != 1:
    print("WARNING: not exactly one root")

if edges == N-1:
    print("Tree property satisfied (E = N-1)")
else:
    print("NOT a tree")
    print("E - (N-1) =", edges-(N-1))
