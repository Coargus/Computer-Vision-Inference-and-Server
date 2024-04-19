"""test script of module."""


def hello_world() -> str:
    """Return Hello World."""
    return "Hello World"


def test_hello_world() -> None:
    """Test hello_world function."""
    assert hello_world() == "Hello World"
