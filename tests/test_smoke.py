import reliability_lab


def test_version_is_exposed() -> None:
    assert isinstance(reliability_lab.__version__, str)
    assert reliability_lab.__version__
