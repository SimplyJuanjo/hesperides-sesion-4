# Sesión 4 · Implementing a Transformer in NumPy

Repositorio de partida del laboratorio del 1 de octubre de 2026. Máster en Finanzas Cuantitativas y Métodos Computacionales, Universidad de las Hespérides.

## Empezar

```text
git clone https://github.com/SimplyJuanjo/hesperides-sesion-4.git
cd hesperides-sesion-4
pip install -r requirements.txt
pytest
```

Todo en rojo: es el punto de partida. Abre tu agente dentro de esta carpeta.

## Qué hay

- `vocabulario.py`: diecisiete palabras con embeddings de cuatro dimensiones hechos a mano, y los pesos de una cabeza de atención.
- `atencion.py`: las funciones de la clase, sin hacer. Cada una tiene sus tests en `tests/`.
- `mapa.py`: dibuja a quién mira cada palabra. `python mapa.py pedí un crédito al banco`
- `explorar.py`: abre en el navegador un visor de lo que hace tu `atencion.py`, paso a paso.
- `gpt2/gpt2_numpy.py`: GPT-2 con tu `atencion.py` (abajo).
- `AGENTS.md`: las reglas para tu agente.

## GPT-2 con tu atención

Con los 22 tests en verde:

```text
pip install safetensors tiktoken
python gpt2/gpt2_numpy.py
python gpt2/gpt2_numpy.py "The capital of Spain is" -n 12
```

La primera vez descarga los pesos de GPT-2 pequeño (548 MB) en `~/.cache/hesperides/gpt2/`. Cada palabra nueva llama 144 veces a tu `atencion`: 12 capas de 12 cabezas.

## Reglas

- Nada de claves, contraseñas ni datos de tu trabajo en esta carpeta.
