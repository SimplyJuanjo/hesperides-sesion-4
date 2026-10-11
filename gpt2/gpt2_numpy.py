"""GPT-2 pequeño (124 millones de parámetros) en NumPy, con tus funciones `embeber` y `atencion`.

    pip install safetensors tiktoken
    python gpt2/gpt2_numpy.py
    python gpt2/gpt2_numpy.py "The capital of Spain is"
    python gpt2/gpt2_numpy.py --atencion otra_atencion.py -n 12

Cada token pasa por 12 capas de 12 cabezas: 144 llamadas a `atencion` por token generado.
Pesos: `model.safetensors` de openai-community/gpt2 (548 MB) en ~/.cache/hesperides/gpt2/; se descargan la primera vez.
Tokenizador: tiktoken, codificación gpt2. Generación voraz: siempre el token más probable.
"""

import argparse
import importlib.util
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken
from safetensors.numpy import load_file

PESOS = Path.home() / ".cache" / "hesperides" / "gpt2" / "model.safetensors"
URL = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
TU_ATENCION = Path(__file__).resolve().parent.parent / "atencion.py"
CAPAS, CABEZAS, D = 12, 12, 768


def cargar_modulo(ruta: Path):
    sys.path.insert(0, str(ruta.parent))
    spec = importlib.util.spec_from_file_location("atencion_de_clase", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def cargar_pesos() -> dict[str, np.ndarray]:
    if not PESOS.exists():
        PESOS.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando GPT-2 (548 MB) en {PESOS}…", flush=True)
        urllib.request.urlretrieve(URL, PESOS)
    return {k.removeprefix("transformer."): v for k, v in load_file(PESOS).items()}


def norma(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    media = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return g * (x - media) / np.sqrt(var + 1e-5) + b


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def bloque(x: np.ndarray, p: dict, capa: int, atencion) -> np.ndarray:
    pre = f"h.{capa}."
    h = norma(x, p[pre + "ln_1.weight"], p[pre + "ln_1.bias"])
    # El sesgo de Q, K y V entra como una fila más de los pesos y una columna de unos en h.
    h = np.hstack([h, np.ones((len(h), 1), dtype=h.dtype)])
    W = np.vstack([p[pre + "attn.c_attn.weight"], p[pre + "attn.c_attn.bias"]])
    dk = D // CABEZAS
    cabezas = []
    for c in range(CABEZAS):
        q, k, v = (slice(inicio + c * dk, inicio + (c + 1) * dk) for inicio in (0, D, 2 * D))
        salida, _ = atencion(h, W[:, q], W[:, k], W[:, v], causal=True)
        cabezas.append(salida)
    x = x + np.hstack(cabezas) @ p[pre + "attn.c_proj.weight"] + p[pre + "attn.c_proj.bias"]
    h = norma(x, p[pre + "ln_2.weight"], p[pre + "ln_2.bias"])
    return x + gelu(h @ p[pre + "mlp.c_fc.weight"] + p[pre + "mlp.c_fc.bias"]) @ p[pre + "mlp.c_proj.weight"] + p[pre + "mlp.c_proj.bias"]


def siguiente(ids: list[int], p: dict, clase) -> int:
    x = clase.embeber(ids, p["wte.weight"]) + clase.embeber(list(range(len(ids))), p["wpe.weight"])
    for capa in range(CAPAS):
        x = bloque(x, p, capa, clase.atencion)
    x = norma(x[-1], p["ln_f.weight"], p["ln_f.bias"])
    return int(np.argmax(p["wte.weight"] @ x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("texto", nargs="?", default="Alan Turing theorized that computers would one day become")
    parser.add_argument("--atencion", type=Path, default=TU_ATENCION, help="archivo con embeber y atencion")
    parser.add_argument("-n", type=int, default=20, help="tokens que generar")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    clase = cargar_modulo(args.atencion.resolve())
    p = cargar_pesos()
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(args.texto)

    print(f"Atención de {args.atencion}\n")
    print(args.texto, end="", flush=True)
    t0 = time.perf_counter()
    try:
        for _ in range(args.n):
            ids.append(siguiente(ids, p, clase))
            print(enc.decode(ids[-1:]), end="", flush=True)
    except NotImplementedError:
        sys.exit("\n\nTe falta `embeber` o `atencion` en tu atencion.py. Ponlos en verde con pytest y vuelve.")
    print(f"\n\n{args.n} tokens en {time.perf_counter() - t0:.1f} s, {args.n * CAPAS * CABEZAS} llamadas a atencion")


if __name__ == "__main__":
    main()
