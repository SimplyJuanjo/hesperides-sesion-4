import numpy as np

from atencion import softmax


def test_softmax_da_probabilidades():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert np.all(p > 0)
    assert np.isclose(p.sum(), 1.0)
    assert np.allclose(p, [0.0900, 0.2447, 0.6652], atol=1e-4)


def test_softmax_aguanta_numeros_grandes():
    p = softmax(np.array([1000.0, 1001.0]))
    assert not np.any(np.isnan(p))
    assert np.allclose(p, [0.2689, 0.7311], atol=1e-4)


def test_softmax_no_cambia_si_sumas_lo_mismo_a_todos():
    x = np.array([0.5, -1.0, 2.0])
    assert np.allclose(softmax(x), softmax(x + 7.0))


def test_softmax_va_por_filas():
    x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    p = softmax(x)
    assert p.shape == (2, 3)
    assert np.allclose(p.sum(axis=1), [1.0, 1.0])
    assert np.allclose(p[1], [1 / 3, 1 / 3, 1 / 3])


def test_softmax_da_peso_cero_a_menos_infinito():
    p = softmax(np.array([0.0, -np.inf, 0.0]))
    assert np.allclose(p, [0.5, 0.0, 0.5])
