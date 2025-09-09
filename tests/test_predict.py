def test_transformer_main_import():
    import importlib
    m = importlib.import_module('main')
    assert hasattr(m, 'main')
