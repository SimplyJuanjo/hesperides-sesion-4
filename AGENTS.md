# AGENTS.md

## Qué es esto

Una cabeza de atención en NumPy, para entender qué pasa dentro de un LLM. Es para aprender: quien lo usa tiene que entender cada línea.

## Reglas

- Usa solo NumPy.
- Deja como están `tests/`, `vocabulario.py` y las funciones de `atencion.py` que ya vienen escritas.
- Una frase de T palabras es una matriz de T filas. En `embeber`, `softmax`, `atencion` y `mascara_causal`, operaciones de matrices, sin bucles sobre las palabras.

## Cómo trabajar

- Un peldaño cada vez: implementa solo la función que te pidan y ejecuta su test, por ejemplo `pytest tests/test_1_embeddings.py`.
- Antes de escribir código, di la forma (filas, columnas) de cada matriz que vas a crear, y espera.
- Después, explica cada línea en una frase, para alguien que no ha usado NumPy.
