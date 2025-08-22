def test_import_transformer():
    import importlib
    mod = importlib.import_module("fks_transformer.main")
    assert mod is not None
