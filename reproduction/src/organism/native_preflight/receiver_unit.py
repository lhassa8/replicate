"""Render the owner-installed receiver service for the combined device profile.

The receiving substrate retains its privileged host-management duties. Its
process tree shares the device slice and private network with the application
and checker. Rendering does not provision, launch, or certify a Linux device.
"""

from pathlib import Path

from .. import device_resources, device_storage, network_budget
from ..network_namespace import namespace_path
from ..runtime_environment import library_environment
from ..systemd_supervision import literal_path, quoted, safe_path


def render(config, *, config_path, python, entrypoint, libraries=(), runtime_seconds=1500):
    if (config.get("resource_slice") != device_resources.SLICE
            or config.get("storage_profile") != device_storage.PROFILE
            or config.get("network_budget") != network_budget.PROFILE):
        raise ValueError("The combined receiver requires all three device profiles")
    device_storage.validate_layout(config)
    namespace = namespace_path(config.get("network_namespace"))
    if type(runtime_seconds) is not int or not 60 <= runtime_seconds <= 96 * 3600:
        raise ValueError("Receiver lifetime is outside the bounded native profile")
    config_path = safe_path(config_path)
    if not config_path.is_relative_to(device_storage.ROOT / "control"):
        raise ValueError("Receiver configuration must be on the control volume")
    for path in (python, entrypoint, *libraries):
        safe_path(path)
    if len(libraries) > 4:
        raise ValueError("Receiver library search paths exceed their bound")
    logs = device_storage.ROOT / "data/logs/receiver"
    environment = {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C.UTF-8",
        "HOME": "/nonexistent", "PYTHONDONTWRITEBYTECODE": "1",
        "AWS_EC2_METADATA_DISABLED": "true", "OTEL_SDK_DISABLED": "true",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        **library_environment(tuple(Path(path) for path in libraries)),
    }
    command = ("/usr/bin/env", "-i", *(key + "=" + value for key, value in environment.items()),
               str(python), "-I", "-B", str(entrypoint), "--config", str(config_path), "serve")
    return "\n".join((
        "[Unit]", "Description=Owner-installed organism receiving substrate",
        "After=local-fs.target",
        "[Service]", "Type=exec", "User=root", "Group=root",
        "Slice=" + device_resources.SLICE, "Delegate=no", "KillMode=control-group",
        "NetworkNamespacePath=" + literal_path(namespace),
        "WorkingDirectory=" + literal_path(device_storage.ROOT / "control"),
        "UMask=0077", "Restart=no", "TimeoutStopSec=20",
        "RuntimeMaxSec=" + str(runtime_seconds), "LimitCORE=0",
        "StandardInput=null", "StandardOutput=append:" + literal_path(logs / "stdout.log"),
        "StandardError=append:" + literal_path(logs / "stderr.log"),
        # ':' disables manager environment expansion; this is never a shell.
        "ExecStart=:" + " ".join(quoted(argument) for argument in command), "",
    ))
