"""QA del visor con funciones ficticias en archivos temporales.

No resuelve los ejercicios ni ejecuta la suite que se entrega al alumnado.
"""

import json
import os
from pathlib import Path

import numpy as np

import explorar
from explorar import execute_stage, revision, run_stage


def archivo_alumno(tmp_path: Path, contenido: str) -> Path:
    archivo = tmp_path / "alumno.py"
    archivo.write_text(contenido, encoding="utf-8")
    return archivo


def ejecutar(archivo: Path, stage="softmax", **entrada):
    return execute_stage({"stage": stage, "phrase": "el banco", **entrada}, archivo)


def archivo_atencion_ficticia(tmp_path: Path, pendiente=False) -> Path:
    # Los outputs son datos a mano. El producto KQ es deliberadamente incorrecto:
    # el visor debe conservarlo, sin sustituirlo por la fórmula esperada.
    funciones = """import numpy as np
def tokenizar(frase):
    ids = []
    for palabra in frase.split():
        ids.append(0)
    return ids
def embeber(ids, tabla):
    return np.array([[1., 0., 0., 0.], [0., 2., 0., 0.]])
def atencion(X, Wq, Wk, Wv, causal=False):
"""
    cuerpo = "    raise NotImplementedError\n" if pendiente else """    Q = np.array([[1., 2.], [3., 4.]])
    K = np.array([[0., 1.], [2., 0.]])
    V = np.array([[.4, .1], [.2, .3]])
    productos = K @ Q.T
    puntuaciones = productos / 2
    if causal:
        mascara = np.array([[True, False], [True, True]])
        puntuaciones = np.where(mascara, puntuaciones, -np.inf)
    pesos = np.array([[1., 0.], [.25, .75]])
    salida = np.array([[.4, .1], [.25, .25]])
    return salida, pesos
"""
    return archivo_alumno(tmp_path, funciones + cuerpo)


def variable_de_hito(respuesta, hito):
    variables = respuesta["frames"][hito["frame"]]["variables"]
    nombre = hito["variable"]
    if nombre and nombre.startswith("retorno["):
        return variables["retorno"]["value"][int(nombre[8:-1])]
    return variables.get(nombre)


def test_pendiente_conserva_fuente_y_no_inventa_resultado(tmp_path):
    texto = "def tokenizar(frase):\n    raise NotImplementedError\n"
    respuesta = ejecutar(archivo_alumno(tmp_path, texto), "tokenizar")

    assert respuesta["status"] == "pending"
    assert respuesta["source"]["text"] == texto
    assert respuesta["result"] is None
    assert respuesta["checks"] == []
    assert respuesta["error"]["function"] == "tokenizar"
    assert respuesta["error"]["line"] == 2
    assert not any("X" in frame["variables"] for frame in respuesta["frames"])


def test_peldanos_independientes_y_dependencia_pendiente(tmp_path):
    archivo = archivo_alumno(tmp_path, """import numpy as np
def tokenizar(frase):
    raise NotImplementedError
def softmax(x):
    reparto = np.array([[0.25, 0.75]])
    return reparto
def embeber(ids, tabla):
    raise AssertionError("No debe ejecutarse sin los IDs del alumno")
""")

    softmax = ejecutar(archivo)
    embeddings = ejecutar(archivo, "embeber")

    assert softmax["status"] == "ok"
    assert softmax["result"]["data"] == [[0.25, 0.75]]
    assert embeddings["status"] == "pending"
    assert embeddings["error"]["function"] == "tokenizar"
    assert not any(frame["function"] == "embeber" for frame in embeddings["frames"])


def test_instantaneas_no_comparten_array_y_se_atribuyen_a_su_linea(tmp_path):
    archivo = archivo_alumno(tmp_path, """def softmax(x):
    paso = x.copy()
    paso[0] = 99
    return paso
""")
    respuesta = ejecutar(archivo, values=[1, 2])
    copia = next(frame for frame in respuesta["frames"] if frame["line"] == 2)
    cambio = next(frame for frame in respuesta["frames"] if frame["line"] == 3)
    retorno = respuesta["frames"][-1]

    assert copia["code"].strip() == "paso = x.copy()"
    assert copia["variables"]["paso"]["data"] == [1.0, 2.0]
    assert cambio["variables"]["paso"]["data"] == [99.0, 2.0]
    assert retorno["line"] == 4
    assert retorno["code"].strip() == "return paso"
    assert retorno["variables"]["retorno"] == respuesta["result"]


def test_error_conserva_la_linea_de_origen_en_funcion_anidada(tmp_path):
    archivo = archivo_alumno(tmp_path, """def auxiliar(x):
    parcial = x.copy()
    raise ValueError("Fallo deliberado del alumno")
def softmax(x):
    return auxiliar(x)
""")
    respuesta = ejecutar(archivo)

    assert respuesta["status"] == "error"
    assert respuesta["result"] is None
    assert respuesta["error"]["type"] == "ValueError"
    assert respuesta["error"]["message"] == "Fallo deliberado del alumno"
    assert respuesta["error"]["function"] == "auxiliar"
    assert respuesta["error"]["line"] == 3
    assert any("parcial" in frame["variables"] for frame in respuesta["frames"])


def test_numeros_no_finitos_siguen_visibles_y_no_rompen_json(tmp_path):
    archivo = archivo_alumno(tmp_path, """def softmax(x):
    invalido = x / 0
    return invalido
""")
    respuesta = run_stage({"stage": "softmax", "phrase": "el banco", "values": [[0, 1, -1]]}, archivo)

    assert respuesta["status"] == "ok"
    assert respuesta["result"]["data"] == [["nan", "inf", "-inf"]]
    assert not next(check["passed"] for check in respuesta["checks"] if check["label"] == "Salida finita")
    assert '"nan"' in json.dumps(respuesta, allow_nan=False)
    assert "RuntimeWarning" in respuesta["warnings"]
    assert "invalido = x / 0" in respuesta["warnings"]


def test_traza_distingue_qk_de_kq_con_valores_ejecutados(tmp_path):
    # Función ficticia: devuelve productos, sin proporcionar atención resuelta.
    plantilla = """import numpy as np
def softmax(x):
    Q = np.array([[1., 2.], [3., 4.]])
    K = np.array([[0., 1.], [2., 0.]])
    puntuaciones = {producto}
    return puntuaciones
"""
    archivo = archivo_alumno(tmp_path, plantilla.format(producto="Q @ K.T"))
    qk = ejecutar(archivo)
    archivo.write_text(plantilla.format(producto="K @ Q.T"), encoding="utf-8")
    kq = ejecutar(archivo)

    assert qk["result"]["data"] == [[2.0, 2.0], [4.0, 6.0]]
    assert kq["result"]["data"] == [[2.0, 4.0], [2.0, 6.0]]
    assert qk["source_hash"] != kq["source_hash"]
    assert next(frame for frame in qk["frames"] if frame["line"] == 5)["variables"]["puntuaciones"] == qk["result"]


def test_recarga_codigo_guardado_con_mismo_tamano_y_fecha(tmp_path):
    archivo = archivo_alumno(tmp_path, "def tokenizar(frase):\n    return [1, 2]\n")
    stat = archivo.stat()
    primera = run_stage({"stage": "tokenizar", "phrase": "el banco"}, archivo)
    archivo.write_text("def tokenizar(frase):\n    return [7, 8]\n", encoding="utf-8")
    os.utime(archivo, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    segunda = run_stage({"stage": "tokenizar", "phrase": "el banco"}, archivo)

    assert archivo.stat().st_size == stat.st_size
    assert archivo.stat().st_mtime_ns == stat.st_mtime_ns
    assert primera["status"] == segunda["status"] == "ok"
    assert primera["result"] == [1, 2]
    assert segunda["result"] == [7, 8]
    assert primera["source_hash"] != segunda["source_hash"]
    assert not (tmp_path / "__pycache__").exists()


def test_mascara_y_argumento_causal_no_se_simulan_en_visor(tmp_path):
    archivo = archivo_alumno(tmp_path, """import numpy as np
def tokenizar(frase):
    return [0, 1, 2]
def embeber(ids, tabla):
    return np.zeros((3, 4))
def mascara_causal(T):
    visible = np.array([[True, False, False], [True, True, False], [True, True, True]])
    return visible
def atencion(X, Wq, Wk, Wv, causal=False):
    if causal:
        reparto = np.array([[1., 0., 0.], [.4, .6, 0.], [.2, .3, .5]])
    else:
        reparto = np.full((3, 3), 1 / 3)
    return X.copy(), reparto
""")
    mascara = ejecutar(archivo, "mascara", phrase="el banco del")
    causal = ejecutar(archivo, "atencion", phrase="el banco del", causal=True)
    sin_mascara = ejecutar(archivo, "atencion", phrase="el banco del", causal=False)

    assert mascara["status"] == causal["status"] == sin_mascara["status"] == "ok"
    assert all(check["passed"] for check in mascara["checks"])
    assert all(check["passed"] for check in causal["checks"])
    assert causal["result"][1]["data"][0] == [1.0, 0.0, 0.0]
    assert np.allclose(sin_mascara["result"][1]["data"][0], [1 / 3] * 3)
    assert any(frame["variables"].get("causal", {}).get("value") is True for frame in causal["frames"])


def test_hitos_de_atencion_omiten_bucle_de_tokens_y_referencian_datos_reales(tmp_path):
    respuesta = ejecutar(archivo_atencion_ficticia(tmp_path), "atencion", causal=True)
    hitos = respuesta["milestones"]

    assert respuesta["status"] == "ok"
    assert hitos
    assert respuesta["frames"][hitos[0]["frame"]]["function"] == "atencion"
    assert hitos[0]["variable"] == "X"
    assert variable_de_hito(respuesta, hitos[0])["data"] == [[1., 0., 0., 0.], [0., 2., 0., 0.]]
    for hito in hitos:
        assert isinstance(hito["id"], str)
        assert isinstance(hito["label"], str)
        assert isinstance(hito["description"], str)
        assert 0 <= hito["frame"] < len(respuesta["frames"])
        assert respuesta["frames"][hito["frame"]]["function"] != "tokenizar"
        if hito["variable"] is not None:
            assert variable_de_hito(respuesta, hito) is not None
    producto = next(hito for hito in hitos if hito["variable"] == "productos")
    assert variable_de_hito(respuesta, producto)["data"] == [[2., 4.], [2., 6.]]
    assert "K @ Q.T" in respuesta["frames"][producto["frame"]]["code"]


def test_hitos_conservan_puntuaciones_antes_y_despues_de_mascara(tmp_path):
    respuesta = ejecutar(archivo_atencion_ficticia(tmp_path), "atencion", causal=True)
    hitos = [hito for hito in respuesta["milestones"] if hito["variable"] == "puntuaciones"]
    estados = [variable_de_hito(respuesta, hito)["data"] for hito in hitos]

    assert [[1., 2.], [1., 3.]] in estados
    assert [[1., "-inf"], [1., 3.]] in estados
    antes = next(hito for hito in hitos if variable_de_hito(respuesta, hito)["data"] == [[1., 2.], [1., 3.]])
    despues = next(hito for hito in hitos if variable_de_hito(respuesta, hito)["data"] == [[1., "-inf"], [1., 3.]])
    assert antes["frame"] < despues["frame"]
    assert "np.where" in respuesta["frames"][despues["frame"]]["code"]


def test_changed_distingue_variables_nuevas_cambio_y_valores_sin_cambios(tmp_path):
    archivo = archivo_alumno(tmp_path, """def softmax(x):
    paso = x.copy()
    copia = paso.copy()
    paso[0] = 99
    return paso
""")
    respuesta = ejecutar(archivo, values=[1, 2])
    copia_inicial = next(frame for frame in respuesta["frames"] if frame["line"] == 2)
    copia_extra = next(frame for frame in respuesta["frames"] if frame["line"] == 3)
    mutacion = next(frame for frame in respuesta["frames"] if frame["line"] == 4)

    assert "paso" in copia_inicial["changed"]
    assert "copia" in copia_extra["changed"]
    assert "paso" not in copia_extra["changed"]
    assert "paso" in mutacion["changed"]
    assert "copia" not in mutacion["changed"]
    assert copia_inicial["variables"]["paso"]["data"] == [1., 2.]
    assert mutacion["variables"]["copia"]["data"] == [1., 2.]


def test_pendiente_no_fabrica_hitos_de_calculos_ausentes(tmp_path):
    respuesta = ejecutar(archivo_atencion_ficticia(tmp_path, pendiente=True), "atencion")

    assert respuesta["status"] == "pending"
    assert respuesta["result"] is None
    assert respuesta["error"]["function"] == "atencion"
    for frame in respuesta["frames"]:
        assert not {"Q", "K", "V", "productos", "puntuaciones", "pesos", "salida"}.intersection(frame["variables"])
    for hito in respuesta["milestones"]:
        assert 0 <= hito["frame"] < len(respuesta["frames"])
        if hito["variable"] is not None:
            assert variable_de_hito(respuesta, hito) is not None
    ultimo = respuesta["frames"][-1]
    assert ultimo["function"] == "atencion"
    assert ultimo["line"] == respuesta["error"]["line"]


def test_nombre_libre_conserva_fallback_a_captura_sin_inventar_intermediarios(tmp_path):
    archivo = archivo_alumno(tmp_path, """import numpy as np
def softmax(x):
    otro = np.array([[0.1, 0.9]])
    return otro
""")
    respuesta = ejecutar(archivo)

    assert respuesta["status"] == "ok"
    assert respuesta["milestones"]
    for hito in respuesta["milestones"]:
        assert respuesta["frames"][hito["frame"]]["function"] == "softmax"
        if hito["variable"] is not None:
            assert variable_de_hito(respuesta, hito) is not None
    assert not any({"e", "denominador", "exponenciales"}.intersection(frame["variables"]) for frame in respuesta["frames"])
    final = respuesta["milestones"][-1]
    assert variable_de_hito(respuesta, final) == respuesta["result"]


def test_ultimo_hito_contexto_es_vector_devuelto_real_y_no_salida_atencion(tmp_path):
    archivo = archivo_alumno(tmp_path, """import numpy as np
def contextualizar(frase, causal=False):
    original = np.array([[1., 0., 0., 0.], [0., 0., 0., 1.]])
    aporte = np.array([[.1, .2, 0., 0.], [.8, .1, 0., 0.]])
    return ['el', 'banco'], original + aporte, np.eye(2)
""")
    respuesta = ejecutar(archivo, "contexto")
    final = respuesta["milestones"][-1]

    assert respuesta["status"] == "ok"
    assert final["variable"] == "retorno[1]"
    assert respuesta["frames"][final["frame"]]["function"] == "contextualizar"
    assert variable_de_hito(respuesta, final) == respuesta["result"][1]
    assert variable_de_hito(respuesta, final)["data"] == [[1.1, .2, 0., 0.], [.8, .1, 0., 1.]]


def test_revision_detecta_contenido_con_misma_fecha_sin_importar_alumno(tmp_path, monkeypatch):
    fuente = archivo_alumno(tmp_path, 'raise RuntimeError("No ejecutar 1")\n')
    interfaz = tmp_path / "index.html"
    interfaz.write_text("<head>Interfaz 1</head>", encoding="utf-8")
    servidor = tmp_path / "servidor.py"
    servidor.write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(explorar, "__file__", str(servidor))
    fechas = {archivo: archivo.stat() for archivo in (fuente, interfaz, servidor)}
    primera = revision(fuente, interfaz)
    en_memoria = primera["server_hash"]

    for archivo in (fuente, interfaz, servidor):
        archivo.write_text(archivo.read_text(encoding="utf-8").replace("1", "2"), encoding="utf-8")
        anterior = fechas[archivo]
        os.utime(archivo, ns=(anterior.st_atime_ns, anterior.st_mtime_ns))
        assert archivo.stat().st_size == anterior.st_size
        assert archivo.stat().st_mtime_ns == anterior.st_mtime_ns

    segunda = revision(fuente, interfaz, running_server_hash=en_memoria)
    for clave in ("source_hash", "ui_hash", "server_hash"):
        assert primera[clave] != segunda[clave]
        assert len(segunda[clave]) == 64
    assert primera["running_server_hash"] == primera["server_hash"]
    assert segunda["running_server_hash"] == en_memoria
    assert segunda["running_server_hash"] != segunda["server_hash"]
    assert not (tmp_path / "__pycache__").exists()
