"""Mac authoring helper: require a real clone, then verify independent files.

This helper is not carried by the organism. Linux delivery and preparation
continue to copy the complete payload and use their existing resource checks.
"""

import ctypes
import errno
import hashlib
import os
from pathlib import Path
import platform
import stat


# Apple copyfile.h and copyfile(3), from the installed macOS SDK.
COPYFILE_CLONE_FORCE = 1 << 25
COPYFILE_EXCL = 1 << 17
COPYFILE_NOFOLLOW = (1 << 18) | (1 << 19)
COPYFILE_STATE_WAS_CLONED = 10


def _copyfile_clone(source, target):
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    library.copyfile_state_alloc.argtypes = []
    library.copyfile_state_alloc.restype = ctypes.c_void_p
    library.copyfile_state_free.argtypes = [ctypes.c_void_p]
    library.copyfile_state_free.restype = ctypes.c_int
    library.copyfile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint32]
    library.copyfile.restype = ctypes.c_int
    library.copyfile_state_get.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    library.copyfile_state_get.restype = ctypes.c_int
    state = library.copyfile_state_alloc()
    if not state:
        raise OSError(errno.ENOMEM, "Cannot allocate copyfile state")
    try:
        # CLONE_FORCE implies exclusive destination creation and returns an
        # error when cloning is unsupported; it never falls back to byte copy.
        if library.copyfile(os.fsencode(source), os.fsencode(target), state,
                            COPYFILE_CLONE_FORCE | COPYFILE_EXCL | COPYFILE_NOFOLLOW) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(target))
        cloned = ctypes.c_bool()
        if library.copyfile_state_get(state, COPYFILE_STATE_WAS_CLONED, ctypes.byref(cloned)) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(target))
        if not cloned.value:
            raise ValueError("The OS did not confirm a cloned file")
    finally:
        library.copyfile_state_free(state)


def _hash_open_file(handle):
    value, count = hashlib.sha256(), 0
    for block in iter(lambda: handle.read(1024**2), b""):
        value.update(block)
        count += len(block)
    return {"bytes": count, "sha256": value.hexdigest()}


def _unchanged(first, second):
    return all(getattr(first, key) == getattr(second, key)
               for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))


def verified_clone(source, target, expected):
    if platform.system() != "Darwin":
        raise ValueError("Mac cloning is an authoring option only")
    source, target = Path(source), Path(target)
    if os.path.lexists(target):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(target))
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as original:
        before = os.fstat(original.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected["bytes"]:
            raise ValueError("Pinned clone input type or length differs")
        if _hash_open_file(original) != expected or not _unchanged(before, os.fstat(original.fileno())):
            raise ValueError("Pinned clone input hash or metadata changed")
        target.parent.mkdir(parents=True, exist_ok=True)
        _copyfile_clone(source, target)
        copied_fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(copied_fd, "rb") as copied:
            info = os.fstat(copied.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or (info.st_dev, info.st_ino) == (before.st_dev, before.st_ino)):
                raise ValueError("Cloned payload must be a separate regular file")
            if _hash_open_file(copied) != expected:
                raise ValueError("Cloned payload hash or length differs")
            if not _unchanged(before, os.fstat(original.fileno())):
                raise ValueError("Pinned clone input changed during cloning")
            os.fchmod(copied.fileno(), 0o600)
            os.fsync(copied.fileno())
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return {"method": "mac_copyfile_clone_force", "os_clone_confirmed": True,
            "source_and_target_hash_verified": True, "distinct_file_identity": True,
            "source_inode": before.st_ino, "target_inode": info.st_ino,
            "target_link_count": info.st_nlink, **expected}
