def saludar(nombre: str = "Mundo") -> str:
    """Saluda al usuario en español."""
    return f"¡Hola, {nombre}!"


if __name__ == "__main__":
    print(saludar())
