"""Fail fast if a unit test attempts a live service or network call."""

from unittest.mock import patch

_PATCHERS = []


def _blocked(*_args, **_kwargs):
    raise AssertionError(
        "network access is disabled in unit tests; mock the service boundary"
    )


def install_offline_guards() -> None:
    """Install process-wide guards once; individual tests may patch over them."""
    if _PATCHERS:
        return

    targets = (
        "socket.socket.connect",
        "socket.socket.connect_ex",
        "e2b_swebench.templates._open_registry",
        "e2b_swebench.templates.Template",
        "e2b_swebench.templates.Sandbox",
        "e2b_swebench.driver.Sandbox",
        "e2b_swebench.dataset.load_dataset",
        "e2b.AsyncSandbox",
    )
    for target in targets:
        patcher = patch(target, side_effect=_blocked)
        patcher.start()
        _PATCHERS.append(patcher)
