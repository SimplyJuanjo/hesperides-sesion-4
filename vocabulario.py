"""Los datos de la sesión 4: un vocabulario de juguete, sus embeddings y los pesos de una cabeza de atención.

Todo está hecho a mano para que se pueda leer. En un modelo de verdad estas matrices salen del entrenamiento
y tienen cientos de dimensiones sin nombre; aquí son cuatro y cada una significa algo.
"""

import numpy as np

DIMENSIONES = ["dinero", "asiento", "significado", "ambigua"]

VOCAB = [
    "me", "en", "el", "un", "al", "del", "de", "la",
    "senté", "parque", "plaza",
    "pedí", "saqué", "crédito", "dinero", "hipoteca",
    "banco",
]

_FILAS = {
    "senté": [0.0, 1.0, 1.0, 0.0],
    "parque": [0.0, 1.0, 1.0, 0.0],
    "plaza": [0.0, 1.0, 1.0, 0.0],
    "pedí": [0.3, 0.0, 1.0, 0.0],
    "saqué": [0.3, 0.0, 1.0, 0.0],
    "crédito": [1.0, 0.0, 1.0, 0.0],
    "dinero": [1.0, 0.0, 1.0, 0.0],
    "hipoteca": [1.0, 0.0, 1.0, 0.0],
    "banco": [0.0, 0.0, 0.0, 1.0],
}

# Una fila por palabra del vocabulario, una columna por dimensión. Las palabras sin fila son de relleno: ceros.
E = np.array([_FILAS.get(palabra, [0.0, 0.0, 0.0, 0.0]) for palabra in VOCAB])

# Pregunta: solo «banco» pregunta algo, y lo que pregunta es «¿quién tiene significado aquí?».
WQ = np.zeros((4, 4))
WQ[3, 2] = 4.0

# Clave: cada palabra se anuncia con su propio vector.
WK = np.eye(4)

# Valor: lo que cada palabra ofrece es su parte de dinero y su parte de asiento.
WV = np.diag([1.0, 1.0, 0.0, 0.0])

FRASES = [
    "pedí un crédito al banco",
    "saqué dinero del banco",
    "me senté en el banco",
    "el banco del parque",
]
