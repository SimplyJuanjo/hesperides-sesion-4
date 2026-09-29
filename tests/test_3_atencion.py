import numpy as np

from atencion import atencion, significado


def datos_de_prueba(T: int = 5, d: int = 4, d_k: int = 3, d_v: int = 2, semilla: int = 0):
    rng = np.random.default_rng(semilla)
    return (rng.normal(size=(T, d)), rng.normal(size=(d, d_k)),
            rng.normal(size=(d, d_k)), rng.normal(size=(d, d_v)))


def test_formas_de_la_salida_y_de_los_pesos():
    X, Wq, Wk, Wv = datos_de_prueba()
    salida, pesos = atencion(X, Wq, Wk, Wv)
    assert pesos.shape == (5, 5)
    assert salida.shape == (5, 2)


def test_cada_fila_de_pesos_es_un_reparto_del_100_por_cien():
    X, Wq, Wk, Wv = datos_de_prueba()
    _, pesos = atencion(X, Wq, Wk, Wv)
    assert np.all(pesos >= 0)
    assert np.allclose(pesos.sum(axis=1), 1.0)


def test_coincide_con_la_formula_hecha_a_mano():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    I = np.eye(2)
    salida, pesos = atencion(X, I, I, I)
    a = np.exp(1 / np.sqrt(2))
    esperado = np.array([[a, 1.0], [1.0, a]]) / (a + 1)
    assert np.allclose(pesos, esperado)
    assert np.allclose(salida, esperado)


def test_con_un_credito_delante_banco_se_acerca_a_dinero():
    s = significado("pedí un crédito al banco")
    assert s["dinero"] > s["parque"]


def test_si_te_sientas_banco_se_acerca_a_parque():
    s = significado("me senté en el banco")
    assert s["parque"] > s["dinero"]


def test_desordenar_la_frase_solo_desordena_la_salida():
    X, Wq, Wk, Wv = datos_de_prueba()
    orden = np.array([3, 0, 4, 1, 2])
    salida, _ = atencion(X, Wq, Wk, Wv)
    salida_desordenada, _ = atencion(X[orden], Wq, Wk, Wv)
    assert np.allclose(salida_desordenada, salida[orden])
