"""Visor local de la ejecución del alumno: python explorar.py.

No implementa las funciones del laboratorio. Captura sus variables al ejecutar
atencion.py y sirve la interfaz de visor/index.html. Cada ejecución lee el
archivo guardado, en un proceso nuevo, sin modificarlo ni cargar una solución.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import types
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DEFAULT_SOURCE = BASE / "atencion.py"
UI_PATH = BASE / "visor" / "index.html"
MAX_FRAMES = 350
STAGES = [
    {"id": "tokenizar", "label": "Palabras e IDs", "question": "¿Qué números representan esta frase?",
     "prompt": "Implementa solo tokenizar. Antes de escribir, explica qué devolverá y espera. Después ejecuta sus tests y explica cada línea. No cambies los tests."},
    {"id": "embeber", "label": "Embeddings", "question": "¿Cuántas filas y columnas tendrá X?",
     "prompt": "Implementa solo embeber. Antes de escribir, di la forma de la entrada y de la salida y espera. Usa NumPy sin bucles sobre las palabras. Ejecuta los tests del peldaño 1 y explica cada línea."},
    {"id": "softmax", "label": "Softmax", "question": "¿Cada fila suma 1, también con valores grandes?",
     "prompt": "Implementa solo softmax. Antes de escribir, explica las formas y el eje sobre el que normalizas y espera. Tiene que aguantar valores grandes y -inf. Ejecuta los tests del peldaño 2 y explica cada línea."},
    {"id": "atencion", "label": "Atención", "question": "¿De dónde sale una celda del reparto?",
     "prompt": "Implementa solo atencion sin máscara por ahora. Antes de escribir, di las formas de X, Q, K, V, las puntuaciones, los pesos y la salida y espera. Deja los pasos intermedios en variables para poder inspeccionarlos. Ejecuta los tests del peldaño 3 y explica cada línea."},
    {"id": "mascara", "label": "Máscara", "question": "¿Qué parejas de palabras pueden verse?",
     "prompt": "Implementa mascara_causal y añade su uso en atencion cuando causal=True. Antes de escribir, explica la forma, la diagonal y dónde se aplica y espera. Ejecuta los tests del peldaño 4 y explica cada línea."},
    {"id": "contexto", "label": "En contexto", "question": "¿Qué aporta la atención al vector de banco?",
     "prompt": "Ejecuta los tests y explica contextualizar sin modificarla. Relaciona cada línea con las matrices del visor. Distingue pesos de atención, vector de salida y similitud coseno. No implementes más funciones."},
]


def source_info(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig")
    return {"file": path.name, "text": text}


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def revision(source_path: Path, ui_path: Path = UI_PATH,
             running_server_hash: str | None = None) -> dict:
    """Comprueba el contenido guardado sin importar ni ejecutar al alumno."""
    server_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"source_hash": source_hash(source_info(source_path)["text"]),
            "ui_hash": hashlib.sha256(ui_path.read_bytes()).hexdigest(),
            "server_hash": server_hash,
            "running_server_hash": running_server_hash or server_hash}


def encode(value):
    """JSON válido también para NaN, infinito, booleanos y matrices NumPy."""
    if isinstance(value, np.ndarray):
        return {"kind": "array", "shape": list(value.shape), "dtype": str(value.dtype),
                "data": encode(value.tolist())}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    return str(value)


class Capture:
    """Instantáneas después de cada línea del archivo del alumno, sin hooks."""

    def __init__(self, path: Path, text: str):
        self.path = str(path.resolve())
        self.lines = text.splitlines()
        self.frames = []
        self.previous = {}
        self.error = None
        self.active = {}
        self.unwinding = set()

    def snapshot(self, frame, line, returned=None, has_return=False, event="line"):
        # Se serializa ahora: cambios posteriores de un array no alteran la traza.
        variables = dict(self.active)
        for name, value in frame.f_locals.items():
            if name.startswith("_") or isinstance(value, (types.ModuleType, types.FunctionType)):
                continue
            if isinstance(value, np.ndarray):
                if value.size <= 2500:
                    variables[name] = encode(value)
            elif isinstance(value, (list, tuple, dict, str, bool, int, float, np.generic)):
                variables[name] = {"kind": "value", "value": encode(value)}
        if has_return:
            variables["retorno"] = encode(returned) if isinstance(returned, np.ndarray) else {
                "kind": "value", "value": encode(returned)}
        changed = [name for name, value in variables.items()
                   if name not in self.active or value != self.active[name]]
        self.active = variables
        self.frames.append({"function": frame.f_code.co_name, "line": line,
                            "code": self.lines[line - 1] if 0 < line <= len(self.lines) else "",
                            "variables": variables, "changed": changed, "event": event,
                            "locals": list(frame.f_locals)})
        if len(self.frames) >= MAX_FRAMES:
            raise RuntimeError("Demasiados pasos para el visor. Usa operaciones NumPy sin bucles sobre las palabras.")

    def __call__(self, frame, event, arg):
        if frame.f_code.co_filename != self.path:
            return None
        key = id(frame)
        if event == "call":
            # Comprehensions no aportan nuevos peldaños y multiplican la traza.
            if frame.f_code.co_name.startswith("<"):
                return None
            self.previous[key] = None
            # Las entradas se capturan al entrar, antes de calcular Q o cualquier salida.
            self.snapshot(frame, frame.f_lineno, event="call")
        elif event == "line":
            self.unwinding.discard(key)
            previous = self.previous.get(key)
            if previous is not None and previous != frame.f_lineno:
                self.snapshot(frame, previous)
            self.previous[key] = frame.f_lineno
        elif event == "exception":
            self.unwinding.add(key)
            error_type, error, _ = arg
            self.error = {"type": error_type.__name__, "message": str(error),
                          "line": frame.f_lineno, "function": frame.f_code.co_name}
            self.snapshot(frame, self.previous.get(key) or frame.f_lineno, event="exception")
        elif event == "return":
            if key not in self.unwinding:
                self.snapshot(frame, self.previous.get(key) or frame.f_lineno, arg, True, "return")
            self.unwinding.discard(key)
            self.previous.pop(key, None)
        return self


def milestones(frames: list[dict], stage: str) -> list[dict]:
    """Selecciona pasos de la traza real; nunca reconstruye resultados ausentes."""
    functions = {"tokenizar": {"tokenizar"}, "embeber": {"embeber"},
                 "softmax": {"softmax"}, "mascara": {"mascara_causal"},
                 "atencion": {"atencion", "softmax", "mascara_causal"},
                 "contexto": {"atencion", "softmax", "mascara_causal", "contextualizar"}}
    labels = {
        "X": ("X", "Vectores que entran en la cabeza de atención."),
        "Q": ("Q", "Preguntas calculadas por tu código."),
        "K": ("K", "Claves calculadas por tu código."),
        "V": ("V", "Valores calculados por tu código."),
        "productos": ("Productos", "Producto ejecutado antes del escalado."),
        "puntuaciones": ("Puntuaciones", "Puntuaciones capturadas en este paso."),
        "mascara": ("Máscara", "Parejas permitidas por la máscara de tu función."),
        "m": ("Máximo", "Máximo por fila usado para estabilizar softmax."),
        "e": ("Exponenciales", "Numeradores calculados por tu código."),
        "denominador": ("Suma por fila", "Denominador capturado para normalizar cada fila."),
        "pesos": ("Pesos", "Reparto que devuelve el softmax ejecutado."),
        "salida": ("Salida", "Aporte de la atención calculado por tu código."),
        "x": ("Entrada", "Valores recibidos por la función."),
    }
    hits = []

    def add(index, name, label, description):
        hits.append({"id": f"{index}:{name}", "label": label, "frame": index,
                     "variable": name, "description": description})

    has_attention = any(frame["function"] == "atencion" for frame in frames)
    for index, frame in enumerate(frames):
        function = frame["function"]
        if function not in functions.get(stage, set()):
            continue
        variables, local_names = frame["variables"], frame["locals"]
        event = frame["event"]
        if function == "contextualizar" and has_attention and event != "return":
            continue
        if event == "call":
            names = ["X"] if function == "atencion" else ["x"] if stage == "softmax" else []
        else:
            names = [name for name in frame["changed"] if name in local_names]
        for name in names:
            value = variables.get(name, {})
            if value.get("kind") != "array":
                continue
            if name in {"E", "tabla", "Wq", "Wk", "Wv", "WQ", "WK", "WV"}:
                continue
            if function == "softmax" and stage != "softmax" and name == "x":
                continue
            label, description = labels.get(name, (name, f"Variable {name} capturada de tu ejecución."))
            if name == "puntuaciones" and "-inf" in str(value["data"]):
                label, description = "Puntuaciones con máscara", "Puntuaciones reales después de bloquear posiciones."
            add(index, name, label, description)
        if event != "return":
            continue
        returned = variables.get("retorno", {})
        if function == "contextualizar":
            values = returned.get("value", [])
            if isinstance(values, list) and len(values) > 1 and isinstance(values[1], dict) and values[1].get("kind") == "array":
                add(index, "retorno[1]", "En contexto", "Vector final devuelto por contextualizar; compara X + salida.")
        elif function == "atencion":
            values = returned.get("value", [])
            if isinstance(values, list):
                for position, name, label in [(0, "salida", "Salida devuelta"), (1, "pesos", "Pesos devueltos")]:
                    if name not in local_names and len(values) > position and isinstance(values[position], dict) and values[position].get("kind") == "array":
                        add(index, f"retorno[{position}]", label, "Matriz del retorno real de tu función.")
        elif stage in {"tokenizar", "embeber", "softmax", "mascara"}:
            add(index, "retorno", "Retorno", "Resultado devuelto por tu función.")
        elif function == "mascara_causal" and returned.get("kind") == "array":
            add(index, "retorno", "Máscara", "Máscara devuelta por tu función.")
    return hits


def _words(phrase: str) -> list[str]:
    # Etiquetas del visor; los IDs y los embeddings siempre los calcula el alumno.
    return re.findall(r"\w+", phrase.lower())


def _validate(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("La ejecución necesita un objeto JSON.")
    stage = payload.get("stage", "tokenizar")
    if stage not in {s["id"] for s in STAGES}:
        raise ValueError("Peldaño desconocido.")
    phrase = payload.get("phrase", "pedí un crédito al banco")
    if not isinstance(phrase, str) or not 1 <= len(_words(phrase)) <= 20:
        raise ValueError("Escribe una frase de 1 a 20 palabras del vocabulario.")
    strength = float(payload.get("strength", 4))
    if not math.isfinite(strength) or not 0 <= strength <= 8:
        raise ValueError("La fuerza de la pregunta debe estar entre 0 y 8.")
    if not isinstance(payload.get("causal", False), bool):
        raise ValueError("La máscara debe ser true o false.")
    return {**payload, "stage": stage, "phrase": phrase, "strength": strength,
            "causal": payload.get("causal", False)}


def _checks(stage: str, result, count: int, causal: bool) -> list[dict]:
    checks = []

    def add(label, passed):
        checks.append({"label": label, "passed": bool(passed)})

    if stage == "tokenizar":
        add("Un ID por palabra", isinstance(result, (list, np.ndarray)) and len(result) == count)
    elif stage in {"embeber", "softmax", "mascara"}:
        add("Devuelve una matriz NumPy", isinstance(result, np.ndarray))
        if not isinstance(result, np.ndarray):
            return checks
        if stage == "embeber":
            add("Una fila por palabra", result.ndim == 2 and result.shape[0] == count)
            add("Salida finita", np.isfinite(result).all())
        elif stage == "softmax":
            add("Salida finita", np.isfinite(result).all())
            add("Pesos no negativos", (result >= 0).all())
            add("Vector o matriz", 1 <= result.ndim <= 2)
            if result.ndim:
                add("Cada fila suma 1", np.allclose(result.sum(axis=-1), 1))
        else:
            add("Forma T × T", result.shape == (count, count))
            add("Diagonal y pasado permitidos", result.shape == (count, count)
                and all(result[i, j] == (j <= i) for i in range(count) for j in range(count)))
    elif stage in {"atencion", "contexto"}:
        expected_count = 2 if stage == "atencion" else 3
        valid = isinstance(result, (list, tuple)) and len(result) == expected_count
        add("Devuelve salida y pesos" if stage == "atencion" else "Devuelve palabras, vectores y pesos", valid)
        if not valid:
            return checks
        weights = result[1] if stage == "atencion" else result[2]
        output = result[0] if stage == "atencion" else result[1]
        if isinstance(weights, np.ndarray):
            add("Pesos de forma T × T", weights.shape == (count, count))
            add("Pesos finitos y no negativos", np.isfinite(weights).all() and (weights >= 0).all())
            add("Cada fila de pesos suma 1", np.allclose(weights.sum(axis=-1), 1))
            if causal and weights.ndim == 2:
                add("Sin atención al futuro", np.allclose(np.triu(weights, k=1), 0))
        if isinstance(output, np.ndarray):
            add("Salida finita", np.isfinite(output).all())
    return checks


def execute_stage(payload: dict, source_path: Path = DEFAULT_SOURCE) -> dict:
    """Ejecuta un peldaño y devuelve su traza. No escribe archivos."""
    source_path = Path(source_path).resolve()
    source = source_info(source_path)
    result = {"status": "error", "stage": payload.get("stage", "tokenizar"),
              "source": source, "source_hash": source_hash(source["text"]),
              "frames": [], "milestones": [], "result": None, "error": None, "checks": [], "words": [], "inputs": {}}
    captured = Capture(source_path, source["text"])
    old_trace, old_path = sys.gettrace(), sys.path[:]
    output, warnings = io.StringIO(), io.StringIO()
    try:
        payload = _validate(payload)
        stage, phrase = payload["stage"], payload["phrase"]
        words = _words(phrase)
        result.update(stage=stage, words=words, inputs={k: payload[k] for k in ("phrase", "causal", "strength")})
        sys.path[:0] = [str(source_path.parent), str(BASE)]
        module = types.ModuleType("atencion_del_alumno")
        module.__file__ = str(source_path)
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(warnings):
            exec(compile(source["text"], str(source_path), "exec"), module.__dict__)
            # La tabla y los pesos son entradas del ejercicio, nunca soluciones.
            from vocabulario import E, WQ, WK, WV
            table = getattr(module, "E", E)
            wq = getattr(module, "WQ", WQ).copy()
            wq[3, 2] = payload["strength"]
            wk, wv = getattr(module, "WK", WK), getattr(module, "WV", WV)
            module.WQ = wq
            captured.active = {"E": encode(table), "Wq": encode(wq), "Wk": encode(wk), "Wv": encode(wv)}
            sys.settrace(captured)
            with np.errstate(all="warn"):
                if stage == "tokenizar":
                    actual = module.tokenizar(phrase)
                elif stage == "embeber":
                    ids = module.tokenizar(phrase)
                    actual = module.embeber(ids, table)
                elif stage == "softmax":
                    values = payload.get("values", [[1, 2, 3], [1, 1, 1]])
                    x = np.asarray(values, dtype=float)
                    if not 1 <= x.ndim <= 2 or x.size == 0 or x.size > 100:
                        raise ValueError("Softmax acepta un vector o una matriz de hasta 100 valores.")
                    actual = module.softmax(x)
                elif stage == "mascara":
                    actual = module.mascara_causal(len(words))
                elif stage == "atencion":
                    ids = module.tokenizar(phrase)
                    x = module.embeber(ids, table)
                    actual = module.atencion(x, wq, wk, wv, causal=payload["causal"])
                else:
                    actual = module.contextualizar(phrase, causal=payload["causal"])
            sys.settrace(old_trace)
        result.update(status="ok", result=encode(actual))
        try:
            result["checks"] = _checks(stage, actual, len(words), payload["causal"])
        except (TypeError, ValueError, IndexError):
            result["checks"] = [{"label": "Salida compatible con el peldaño", "passed": False}]
    except Exception as error:
        sys.settrace(old_trace)
        location = captured.error or {"line": getattr(error, "lineno", None), "function": None}
        traceback = error.__traceback__
        while traceback:
            if traceback.tb_frame.f_code.co_filename == str(source_path):
                location = {"line": traceback.tb_lineno, "function": traceback.tb_frame.f_code.co_name}
            traceback = traceback.tb_next
        result["error"] = {**location, "type": type(error).__name__, "message": str(error)}
        result["status"] = "pending" if isinstance(error, NotImplementedError) else "error"
        if result["status"] == "pending":
            result["error"]["message"] = "Esta función sigue pendiente. Complétala en atencion.py, guarda y vuelve a ejecutar."
    finally:
        sys.settrace(old_trace)
        sys.path[:] = old_path
    result["frames"] = captured.frames
    result["milestones"] = milestones(captured.frames, result["stage"])
    result["stdout"], result["warnings"] = output.getvalue()[-4000:], warnings.getvalue()[-4000:]
    return result


def run_stage(payload: dict, source_path: Path = DEFAULT_SOURCE, timeout: float = 8) -> dict:
    """Un proceso nuevo por ejecución: recarga código y acota errores/bucles."""
    source_path = Path(source_path).resolve()
    try:
        worker = subprocess.run([sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
                                 "--worker", "--archivo", str(source_path)],
                                input=json.dumps(payload, ensure_ascii=False), capture_output=True,
                                text=True, encoding="utf-8", timeout=timeout, cwd=BASE)
        if worker.returncode != 0:
            raise RuntimeError(worker.stderr[-1500:] or "El proceso de ejecución se ha detenido.")
        return json.loads(worker.stdout)
    except (subprocess.TimeoutExpired, RuntimeError, ValueError) as error:
        message = "La ejecución superó 8 segundos. Revisa si hay un bucle o una operación demasiado grande." if isinstance(error, subprocess.TimeoutExpired) else str(error)
        source = source_info(source_path)
        return {"status": "error", "stage": payload.get("stage", "tokenizar"), "source": source,
                "source_hash": source_hash(source["text"]), "frames": [], "milestones": [],
                "result": None, "words": _words(payload.get("phrase", "")), "checks": [],
                "inputs": payload, "error": {"type": type(error).__name__, "message": message, "line": None, "function": None}}


def info(source_path: Path) -> dict:
    from vocabulario import DIMENSIONES, FRASES, VOCAB
    source = source_info(source_path)
    return {"phrase": FRASES[0], "phrases": FRASES, "vocabulary": VOCAB, "dimensions": DIMENSIONES,
            "stages": STAGES, "source": source, "source_hash": source_hash(source["text"]),
            "external": [
                {"label": "Transformer Explainer", "url": "https://poloclub.github.io/transformer-explainer/", "description": "Reconoce atención y predicción en GPT-2 con pesos aprendidos."},
                {"label": "LLM Visualization", "url": "https://bbycroft.net/llm", "description": "Sitúa nuestra cabeza dentro del bloque y del modelo completo."},
            ]}


def serve(source_path: Path, port: int, open_browser: bool):
    running_server_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    class Handler(BaseHTTPRequestHandler):
        def send(self, data, status=200, content_type="application/json; charset=utf-8"):
            body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/info":
                details = info(source_path)
                current = revision(source_path, running_server_hash=running_server_hash)
                # El hash debe describir el texto enviado aunque se guarde durante esta petición.
                current["source_hash"] = details["source_hash"]
                self.send({**details, **current})
            elif self.path in ("/api/revision", "/api/version"):
                self.send(revision(source_path, running_server_hash=running_server_hash))
            elif self.path in ("/", "/index.html"):
                html = UI_PATH.read_bytes()
                current = revision(source_path, running_server_hash=running_server_hash)
                current["ui_hash"] = hashlib.sha256(html).hexdigest()
                bootstrap = ("<script>window.__VISOR_REVISION__=" + json.dumps(current) + "</script>").encode()
                self.send(html.replace(b"</head>", bootstrap + b"</head>", 1), content_type="text/html; charset=utf-8")
            elif self.path == "/favicon.ico":
                self.send(b"", 204, "image/x-icon")
            else:
                self.send({"error": "No existe esta ruta."}, 404)

        def do_POST(self):
            if self.path != "/api/run":
                self.send({"error": "No existe esta ruta."}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Petición vacía o demasiado grande.")
                payload = json.loads(self.rfile.read(length).decode())
                _validate(payload)
                self.send(run_stage(payload, source_path))
            except (ValueError, TypeError) as error:
                self.send({"error": str(error)}, 400)

        def log_message(self, format, *args):
            if args and str(args[1] if len(args) > 1 else "").startswith("5"):
                super().log_message(format, *args)

    try:
        server = HTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        server = HTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"\nVisor de atención: {url}\nArchivo del alumno: {source_path}\nGuarda tu código y pulsa Ejecutar. Ctrl+C para cerrar.\n", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nVisor cerrado.")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puerto", type=int, default=8764)
    parser.add_argument("--sin-abrir", action="store_true", help="No abrir el navegador automáticamente.")
    parser.add_argument("--archivo", type=Path, default=DEFAULT_SOURCE, help="Archivo del alumno que se va a ejecutar.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    args.archivo = args.archivo.resolve()
    if not args.archivo.is_file():
        parser.error(f"No existe {args.archivo}")
    if args.worker:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(execute_stage(json.load(sys.stdin), args.archivo), ensure_ascii=False, allow_nan=False))
    else:
        serve(args.archivo, args.puerto, not args.sin_abrir)


if __name__ == "__main__":
    main()
