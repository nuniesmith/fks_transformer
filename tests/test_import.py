def test_import_transformer():
    import importlib, sys, pathlib
    try:
        mod = importlib.import_module("fks_transformer.main")
    except ModuleNotFoundError:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
        mod = importlib.import_module("main")
    assert hasattr(mod, "main")
