"""Registry fault-tolerance + LocateAnything registration tests."""
import importlib

from src.engine.backends import get_backend, registered_backend_ids


def test_locateanything_backend_registered():
    ids = registered_backend_ids()
    assert "locateanything" in ids
    backend = get_backend("locateanything")
    assert backend.backend_id == "locateanything"


def test_registration_failure_does_not_break_ultralytics(monkeypatch):
    """If LA registration raises, Ultralytics must still be available."""
    import src.engine.backends.registry as reg

    # Re-run registration with a forced failure on the LA import path.
    orig_register = reg.register_backend

    def boom_backend():
        raise RuntimeError("LA exploded")

    # Patch the lazy import to raise inside _register_builtin_backends.
    monkeypatch.setattr(
        "src.engine.backends.locateanything.LocateAnythingBackend",
        boom_backend,
        raising=False,
    )

    # Clear and re-register; should not propagate the LA failure.
    reg._BACKENDS.clear()
    reg._register_builtin_backends()

    ids = reg.registered_backend_ids()
    assert "ultralytics" in ids

    # Restore real registration so other tests see LA again.
    reg._BACKENDS.clear()
    importlib.reload(reg)
    assert "ultralytics" in reg.registered_backend_ids()
