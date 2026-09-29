import numpy as np

from atencion import atencion, mascara_causal, significado


def test_la_mascara_es_triangular():
    esperado = np.array([
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ])
    assert np.array_equal(mascara_causal(4), esperado)


def test_con_mascara_nadie_mira_al_futuro():
    rng = np.random.default_rng(1)
    X, W = rng.normal(size=(6, 4)), rng.normal(size=(4, 4))
    _, pesos = atencion(X, W, W, W, causal=True)
    assert np.allclose(np.triu(pesos, k=1), 0.0)
    assert np.allclose(pesos.sum(axis=1), 1.0)
    assert np.isclose(pesos[0, 0], 1.0)


def test_cambiar_la_ultima_palabra_no_cambia_las_anteriores():
    rng = np.random.default_rng(2)
    X, W = rng.normal(size=(6, 4)), rng.normal(size=(4, 4))
    alterado = X.copy()
    alterado[-1] *= 3
    salida, _ = atencion(X, W, W, W, causal=True)
    salida_alterada, _ = atencion(alterado, W, W, W, causal=True)
    assert np.allclose(salida[:-1], salida_alterada[:-1])


def test_sin_mascara_el_banco_del_parque_ya_es_un_asiento():
    s = significado("el banco del parque")
    assert s["parque"] > s["dinero"]


def test_con_mascara_el_banco_del_parque_todavia_no_sabe_lo_que_es():
    s = significado("el banco del parque", causal=True)
    assert s["parque"] == s["dinero"]
