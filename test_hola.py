from hola import saludar


def test_saludar_default():
    assert saludar() == "¡Hola, Mundo!"


def test_saludar_con_nombre():
    assert saludar("Roberto") == "¡Hola, Roberto!"
