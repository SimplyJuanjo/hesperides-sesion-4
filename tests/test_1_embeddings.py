import numpy as np
import pytest

from atencion import embeber, tokenizar
from vocabulario import E, VOCAB


def test_tokenizar_devuelve_la_posicion_de_cada_palabra():
    esperado = [VOCAB.index(p) for p in ["pedí", "un", "crédito", "al", "banco"]]
    assert tokenizar("pedí un crédito al banco") == esperado


def test_tokenizar_ignora_mayusculas_y_signos():
    assert tokenizar("Pedí un crédito al banco.") == tokenizar("pedí un crédito al banco")


def test_tokenizar_avisa_de_la_palabra_que_no_conoce():
    with pytest.raises(ValueError, match="hola"):
        tokenizar("hola banco")


def test_embeber_una_fila_por_palabra():
    X = embeber(tokenizar("pedí un crédito al banco"), E)
    assert X.shape == (5, 4)


def test_embeber_cada_fila_es_la_de_su_palabra():
    X = embeber(tokenizar("pedí un crédito al banco"), E)
    assert np.array_equal(X[4], E[VOCAB.index("banco")])
    assert np.array_equal(X[2], E[VOCAB.index("crédito")])


def test_embeber_es_multiplicar_por_la_tabla_una_matriz_de_unos_y_ceros():
    ids = tokenizar("me senté en el banco")
    one_hot = np.eye(len(VOCAB))[ids]
    assert np.allclose(embeber(ids, E), one_hot @ E)
