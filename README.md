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
- `AGENTS.md`: las reglas para tu agente.

## Reglas

- Nada de claves, contraseñas ni datos de tu trabajo en esta carpeta.
