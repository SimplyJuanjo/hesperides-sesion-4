"""Una cabeza de atención en NumPy, pieza a pieza.

Las cinco primeras funciones están sin hacer: son los peldaños de la clase. Los tests de `tests/` dicen
qué tiene que ser verdad para cada una. Las dos últimas ya están escritas y juntan las piezas.

Convención: una frase de T palabras es una matriz de T filas. Cada fila es una palabra.
"""

import sys

import numpy as np

from vocabulario import E, VOCAB, WK, WQ, WV


# ---------- Peldaño 1: de palabras a vectores ----------

def tokenizar(frase: str) -> list[int]:
    """Pasa la frase a minúsculas, quita los signos de puntuación y devuelve la posición de cada palabra en VOCAB.

    Si una palabra no está en VOCAB, lanza ValueError con la palabra en el mensaje.
    """
    raise NotImplementedError


def embeber(ids: list[int], tabla: np.ndarray) -> np.ndarray:
    """Devuelve la matriz X de forma (T, d): la fila i es la fila de `tabla` de la palabra ids[i]."""
    raise NotImplementedError


# ---------- Peldaño 2: softmax ----------

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax por filas (sobre el último eje). Cada fila suma 1.

    Tiene que aguantar valores grandes, como [1000, 1001], y valores -inf, que reciben peso 0.
    """
    raise NotImplementedError


# ---------- Peldaño 3: una cabeza de atención ----------

def atencion(X: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
             causal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Atención de una cabeza.

    Q = X Wq, K = X Wk, V = X Wv. Pesos = softmax(Q Kᵀ / √d_k), con d_k el número de columnas de Wq.
    Salida = Pesos V. Devuelve (salida, pesos): salida de forma (T, columnas de Wv), pesos de forma (T, T).

    Con causal=True, la palabra i solo puede mirar a las palabras 0..i (peldaño 4).
    """
    raise NotImplementedError


# ---------- Peldaño 4: el futuro no se ve ----------

def mascara_causal(T: int) -> np.ndarray:
    """Matriz booleana (T, T): True donde la palabra de la fila puede mirar a la de la columna (la columna ≤ la fila)."""
    raise NotImplementedError


# ---------- Ya escritas: juntan las piezas ----------

def contextualizar(frase: str, causal: bool = False) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Cada palabra sale como su embedding más lo que le trae la atención. Devuelve (palabras, vectores, pesos)."""
    ids = tokenizar(frase)
    X = embeber(ids, E)
    salida, pesos = atencion(X, WQ, WK, WV, causal=causal)
    return [VOCAB[i] for i in ids], X + salida, pesos


def significado(frase: str, palabra: str = "banco", causal: bool = False) -> dict[str, float]:
    """Parecido (coseno) de `palabra`, ya en contexto, con «dinero» y con «parque»."""
    palabras, vectores, _ = contextualizar(frase, causal=causal)
    v = vectores[palabras.index(palabra)]

    def coseno(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    return {ref: round(coseno(v, E[VOCAB.index(ref)]), 3) for ref in ("dinero", "parque")}


if __name__ == "__main__":
    frase = " ".join(sys.argv[1:]) or "pedí un crédito al banco"
    print(f"{frase}:", significado(frase))
