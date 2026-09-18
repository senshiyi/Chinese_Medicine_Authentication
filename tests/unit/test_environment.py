from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_isolated(script: str) -> dict:
    project_root = Path(__file__).resolve().parents[2]
    package_root = project_root / "项目" / "Files"
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(package_root) + (os.pathsep + existing if existing else "")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def test_bootstrap_preflight_is_stdlib_only_and_uses_utf8_internally() -> None:
    payload = _run_isolated(
        "import json; "
        "from herb_authentication.environment import run_bootstrap_preflight; "
        "r=run_bootstrap_preflight(); "
        "print(json.dumps({'python_supported': r.python_supported, "
        "'internal_text_encoding': r.internal_text_encoding, "
        "'imported_heavy_modules': r.imported_heavy_modules}))"
    )

    assert payload["python_supported"] is True
    assert payload["internal_text_encoding"] == "utf-8"
    assert payload["imported_heavy_modules"] == []


def test_dependency_probe_uses_only_the_current_isolated_environment() -> None:
    payload = _run_isolated(
        "import json, sys; "
        "from herb_authentication.environment import run_dependency_probe; "
        "p=run_dependency_probe(); "
        "heavy=sorted({'torch','torchvision','numpy','sklearn'}.intersection(sys.modules)); "
        "print(json.dumps({'ok': p.ok, 'error': p.error, "
        "'kmp_duplicate_lib_ok': p.kmp_duplicate_lib_ok, "
        "'packages': {k: v.path for k, v in p.packages.items()}, "
        "'heavy_after': heavy, 'prefix': sys.prefix}))"
    )

    assert payload["ok"] is True, payload["error"]
    assert payload["kmp_duplicate_lib_ok"] is None
    assert set(payload["packages"]) == {
        "Pillow",
        "numpy",
        "sklearn",
        "torch",
        "torchvision",
    }
    environment_root = Path(payload["prefix"]).resolve()
    for package_path_value in payload["packages"].values():
        package_path = Path(package_path_value).resolve()
        assert package_path.is_relative_to(environment_root), package_path
    assert payload["heavy_after"] == []


def test_data_contract_import_does_not_load_torch_or_torchvision() -> None:
    payload = _run_isolated(
        "import json, sys; "
        "import herb_authentication.data; "
        "print(json.dumps({'torch': 'torch' in sys.modules, "
        "'torchvision': 'torchvision' in sys.modules, "
        "'numpy': 'numpy' in sys.modules}))"
    )

    assert payload == {"torch": False, "torchvision": False, "numpy": False}
