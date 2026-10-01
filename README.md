# Sesión 4 · Implementing a Transformer in NumPy

Repositorio de partida del laboratorio del 1 de octubre de 2026. Máster en Finanzas Cuantitativas y Métodos Computacionales, Universidad de las Hespérides.

## Empezar

```text
git clone https://github.com/SimplyJuanjo/hesperides-sesion-4.git
cd hesperides-sesion-4
pip install -r requirements.txt
pytest
```

El repositorio de partida tiene las cinco funciones pendientes y sus tests en rojo.
Si ya has hecho el laboratorio o un dry-run, tus funciones siguen implementadas:
abrir el visor no restaura la plantilla. Abre tu agente dentro de esta carpeta.

## Qué hay

- `vocabulario.py`: diecisiete palabras con embeddings de cuatro dimensiones hechos a mano, y los pesos de una cabeza de atención.
- `atencion.py`: las funciones de la clase. La plantilla inicial las deja pendientes; cada una tiene sus tests en `tests/`.
- `mapa.py`: dibuja a quién mira cada palabra. `python mapa.py pedí un crédito al banco`
- `AGENTS.md`: las reglas para tu agente.

## Explorar el código y las matrices

```text
python explorar.py
```

Se abre un visor local en el navegador. Mantén también el editor y la terminal abiertos.
En Windows, si `python` no está disponible, usa `py explorar.py`.

El visor **ejecuta tu `atencion.py`** y captura sus variables después de cada línea.
No completa funciones, no modifica archivos y no incluye una solución. Si una función
está pendiente, muestra la línea donde se detiene. Al guardar código en el editor,
el visor actualiza el texto y retira la captura anterior para no mezclar versiones.
Pulsa **Ejecutar mi código** para capturar la nueva ejecución: no necesitas reiniciar
el servidor ni recargar la página.

**Aula** recorre los hitos del cálculo: X, Q, K, V, productos, puntuaciones, máscara,
exponenciales, denominador, pesos y salida, cuando existen en tu código.
**Traza completa** conserva las líneas y los bucles para depurar. Las flechas,
Inicio y Fin cambian de paso; los nombres de los hitos permiten saltar directamente.

Selecciona una fila o una celda: la matriz y el inspector siguen la misma selección.
Puedes descomponer el producto QK, el escalado, el reparto del softmax y la mezcla
de V. En contexto, compara el vector inicial, el aporte de atención y el vector
final devuelto por `contextualizar`. Si tu cálculo difiere de la cuenta esperada,
el visor conserva tus valores y señala la diferencia.

El modo Aula aprovecha todo el ancho y mantiene código, matriz e inspector visibles
en una ventana de 1280 × 720. **A− / A+** ajustan el texto para compartir pantalla.
La tipografía mantiene su tamaño al ampliar la ventana. Las líneas largas del código
se ajustan al panel; en ventanas estrechas, los paneles se colocan uno debajo de otro.
Si cambia el HTML del visor mientras está abierto, se recarga y conserva los
controles; vuelve a ejecutar para obtener una captura. Si cambia `explorar.py`,
el visor avisa de que debes reiniciar el servidor.

Recorre los peldaños: palabras e IDs, embeddings, softmax, atención, máscara y contexto.
Softmax y máscara se pueden explorar por separado aunque atención todavía esté pendiente.
Las matrices de entrada vienen de `vocabulario.py`; las matrices intermedias y las salidas
vienen de tu ejecución. Si escribes toda una operación en una sola expresión, sus
intermediarios no aparecerán: deja variables con nombres para poder inspeccionarlas.

Los controles cambian las entradas de esa ejecución: frase, máscara y fuerza de la
pregunta de «banco» (`WQ[3, 2]`). No reescriben el vocabulario ni los pesos del archivo.
Un peso de atención, una coordenada del vector y una similitud coseno son magnitudes diferentes.

Las comprobaciones del visor son invariantes de esa ejecución. Para los tests del
laboratorio, sigue usando la terminal:

```text
python -m pytest tests/test_1_embeddings.py
python -m pytest tests/test_2_softmax.py
python -m pytest tests/test_3_atencion.py
python -m pytest tests/test_4_mascara.py
```

La interfaz funciona sin servicios externos. Para explorar el modelo completo,
incluye enlaces a [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
y [LLM Visualization](https://bbycroft.net/llm). Esas herramientas necesitan Internet
y utilizan ejemplos diferentes de nuestras matrices didácticas.

## Dry-run desde el punto de partida

1. Abre esta carpeta en el editor y ejecuta `python -m pytest tests`: en la plantilla
   inicial hay 22 fallos esperados; en una copia ya implementada deben pasar.
2. Abre otra terminal y ejecuta `python explorar.py`. Si solo quieres la URL,
   usa `python explorar.py --sin-abrir`.
3. En el visor, elige un peldaño. Predice la forma o el resultado antes de ejecutarlo.
4. Copia el encargo del peldaño a un chat nuevo del agente. Trabaja solo en `atencion.py`.
5. Guarda, ejecuta sus tests y pulsa **Ejecutar mi código** en el visor. Sigue los
   hitos de Aula o las líneas de Traza completa y explica una celda con sus valores reales.
6. Cambia una sola entrada cada vez y predice qué debería cambiar. Prueba la máscara
   con «el banco del parque»: la posición de «banco» no puede leer «parque» posterior.

El visor no necesita comandos de Git. Para terminar, pulsa Ctrl+C en su terminal.
Si el puerto 8764 está ocupado, se elige uno libre y se imprime la URL correcta.

Las pruebas de la herramienta están separadas de los 22 tests de la clase:

```text
python -m pytest pruebas_visor
```

## Reglas

- Nada de claves, contraseñas ni datos de tu trabajo en esta carpeta.
