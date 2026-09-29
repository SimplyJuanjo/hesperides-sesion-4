"""Dibuja a quién mira cada palabra de una frase y guarda el dibujo en mapa.png.

    python mapa.py pedí un crédito al banco
    python mapa.py --causal el banco del parque
"""

import sys

import matplotlib.pyplot as plt

from atencion import contextualizar, significado

plt.switch_backend("Agg")
args = sys.argv[1:]
causal = "--causal" in args
frase = " ".join(a for a in args if a != "--causal") or "pedí un crédito al banco"

palabras, _, pesos = contextualizar(frase, causal=causal)

fig, ax = plt.subplots(figsize=(1.1 * len(palabras) + 2, 1.1 * len(palabras) + 1))
ax.imshow(pesos, cmap="RdPu", vmin=0, vmax=1)
ax.set_xticks(range(len(palabras)), palabras, rotation=30, ha="right")
ax.set_yticks(range(len(palabras)), palabras)
ax.set_xlabel("mira a…")
ax.set_ylabel("la palabra…")
for i in range(len(palabras)):
    for j in range(len(palabras)):
        ax.text(j, i, f"{pesos[i, j]:.2f}", ha="center", va="center", color="white" if pesos[i, j] > 0.5 else "#1C1C53")
ax.set_title(("Con máscara: " if causal else "") + frase)
fig.tight_layout()
fig.savefig("mapa.png", dpi=120)

print(frase, "→ banco se parece a", significado(frase, causal=causal) if "banco" in palabras else "(no hay banco)")
print("Dibujo guardado en mapa.png")
