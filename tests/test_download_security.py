"""Pruebas de la validación de URL antes de descargar con curl."""

import json

import pytest

from mi_bici.config import MIBICI_BASE_URL, REFERENCES_DIR, STATIONS_FILENAME
from mi_bici.dataset import download_file, validate_source_url


def test_manifest_urls_are_allowed() -> None:
    """Todas las fuentes reales del manifiesto pasan la validación."""
    manifest = json.loads((REFERENCES_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    urls = [item["source_url"] for item in manifest["monthly_files"]]

    # Misma URL que arma process_stations() para el catálogo de estaciones.
    urls.append(f"{MIBICI_BASE_URL}/site/assets/files/1118/{STATIONS_FILENAME}")

    assert len(urls) > 1
    for url in urls:
        assert validate_source_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://mibici.net/site/assets/files/1/datos_abiertos_2025_01.csv",
        "https://otro-sitio.com/datos_abiertos_2025_01.csv",
        "https://mibici.net.otro-sitio.com/datos_abiertos_2025_01.csv",
        "file:///etc/passwd",
        "ftp://mibici.net/datos_abiertos_2025_01.csv",
    ],
)
def test_untrusted_urls_are_rejected(url: str) -> None:
    """Se rechaza otro esquema u otro dominio, aunque se parezca al de MiBici."""
    with pytest.raises(ValueError, match="no permitida"):
        validate_source_url(url)


def test_download_rejects_url_before_calling_curl(tmp_path, monkeypatch) -> None:
    """Una URL no permitida falla antes de ejecutar curl y no deja archivos."""

    def fail_if_called(*args, **kwargs):
        raise AssertionError("curl no debe ejecutarse")

    monkeypatch.setattr("mi_bici.dataset.subprocess.run", fail_if_called)

    with pytest.raises(ValueError):
        download_file("http://mibici.net/archivo.csv", tmp_path / "archivo.csv")

    assert list(tmp_path.iterdir()) == []
