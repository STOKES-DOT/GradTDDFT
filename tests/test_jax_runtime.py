from td_graddft import jax_runtime


def test_configure_jax_persistent_cache_ignores_missing_xla_cache_option(
    monkeypatch,
    tmp_path,
):
    updates = []

    def fake_update(name, value):
        if name == "jax_persistent_cache_enable_xla_caches":
            raise AttributeError(f"Unrecognized config option: {name}")
        updates.append((name, value))

    monkeypatch.setattr(jax_runtime.jax.config, "update", fake_update)

    result = jax_runtime.configure_jax_persistent_cache(
        cache_dir=str(tmp_path),
        min_compile_time_secs=0.0,
        min_entry_size_bytes=0,
    )

    assert result == str(tmp_path.resolve())
    assert ("jax_compilation_cache_dir", str(tmp_path.resolve())) in updates
    assert ("jax_enable_compilation_cache", True) in updates
    assert ("jax_persistent_cache_min_compile_time_secs", 0.0) in updates
    assert ("jax_persistent_cache_min_entry_size_bytes", 0) in updates


def test_configure_jax_persistent_cache_initializes_legacy_cache_api(
    monkeypatch,
    tmp_path,
):
    calls = []

    class FakeCompilationCache:
        @staticmethod
        def set_cache_dir(path):
            calls.append(path)

    real_import = jax_runtime.importlib.import_module

    def fake_import_module(name):
        if name == "jax.experimental.compilation_cache.compilation_cache":
            return FakeCompilationCache
        return real_import(name)

    monkeypatch.setattr(jax_runtime.importlib, "import_module", fake_import_module)

    result = jax_runtime.configure_jax_persistent_cache(
        cache_dir=str(tmp_path),
        min_compile_time_secs=0.0,
        min_entry_size_bytes=0,
    )

    assert result == str(tmp_path.resolve())
    assert calls == [str(tmp_path.resolve())]
