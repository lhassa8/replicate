"""Explicit carried-library environment shared by the application and checker."""

from pathlib import Path


def library_environment(paths):
    paths = tuple(Path(path) for path in paths)
    if any(not path.is_absolute() or ".." in path.parts or ":" in str(path)
           or any(ord(char) < 32 or ord(char) == 127 for char in str(path)) for path in paths):
        raise ValueError("Invalid carried library path")
    if not paths:
        return {}
    result = {"LD_LIBRARY_PATH": ":".join(map(str, paths))}
    supplemental = [path for path in paths if path.parts[-2:] == ("supplemental", "lib")]
    if len(supplemental) > 1:
        raise ValueError("Ambiguous carried OpenSSL module directory")
    if supplemental:
        result.update(OPENSSL_CONF="/dev/null", OPENSSL_MODULES=str(supplemental[0]),
                      OPENSSL_ENGINES=str(supplemental[0]))
    return result
