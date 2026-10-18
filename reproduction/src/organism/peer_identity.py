"""Offline lab certificates and pinned peer identity for the TLS transport."""

from datetime import datetime, timedelta, timezone
import hashlib
import ipaddress
import os
from pathlib import Path
import ssl
import stat

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from .enrollment import key_id, public_bytes


PROTOCOL = "organism-peer-v1"


class PeerIdentityError(ValueError):
    pass


def device_hostname(identity):
    return f"d-{identity[:32]}.{identity[32:]}.organism.invalid"


def owner_certificate(owner_key, *, now=None, days=30):
    """Owner preparation only; no CA private key is installed on a device."""
    now = now or datetime.now(timezone.utc)
    if not 1 <= days <= 90:
        raise PeerIdentityError("Certificate validity is outside the lab bound")
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Organism isolated lab owner")])
    return (
        x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(owner_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(True, False, False, False, False, True, True, False, False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(owner_key.public_key()), critical=False)
        .sign(owner_key, algorithm=None)
    )


def device_certificate(owner_key, owner_cert, device_public_key, *, now=None, days=7):
    now = now or datetime.now(timezone.utc)
    if not 1 <= days <= 30:
        raise PeerIdentityError("Device certificate validity is outside the lab bound")
    identity = key_id(public_bytes(device_public_key))
    end = min(now + timedelta(days=days), owner_cert.not_valid_after_utc)
    return (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Organism device " + identity[:16])]))
        .issuer_name(owner_cert.subject).public_key(device_public_key).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5)).not_valid_after(end)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH, ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectAlternativeName([
            x509.UniformResourceIdentifier("urn:organism:device:" + identity),
            x509.DNSName(device_hostname(identity)),
        ]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(device_public_key), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(owner_key.public_key()), critical=False)
        .sign(owner_key, algorithm=None)
    )


def write_identity(directory, device_key, certificate, owner_cert):
    """Write a new local identity directory; never overwrite an existing key."""
    directory = Path(directory).absolute()
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    contents = {
        "key.pem": device_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()),
        "certificate.pem": certificate.public_bytes(serialization.Encoding.PEM),
        "owner.pem": owner_cert.public_bytes(serialization.Encoding.PEM),
    }
    for name, data in contents.items():
        fd = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return directory


def certificate_identity(certificate, policy):
    public = certificate.public_key()
    if not isinstance(public, Ed25519PublicKey):
        raise PeerIdentityError("Peer key type differs from enrollment")
    raw = public_bytes(public)
    identity = key_id(raw)
    if policy.peer_key(identity) != raw:
        raise PeerIdentityError("Peer key differs from enrollment")
    if certificate.extensions.get_extension_for_class(x509.BasicConstraints).value.ca:
        raise PeerIdentityError("A CA certificate cannot act as a device")
    alternatives = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    if alternatives.get_values_for_type(x509.UniformResourceIdentifier) != ["urn:organism:device:" + identity]:
        raise PeerIdentityError("Peer certificate has the wrong device identity")
    if alternatives.get_values_for_type(x509.DNSName) != [device_hostname(identity)]:
        raise PeerIdentityError("Peer certificate has the wrong service identity")
    return identity


def context(policy, identity, directory, *, server=False, owner_uid=None, read_group=None):
    owner_uid = os.geteuid() if owner_uid is None else owner_uid
    directory = Path(directory)
    if read_group is not None:
        entry = directory.lstat()
        if (not stat.S_ISDIR(entry.st_mode) or entry.st_uid != owner_uid or entry.st_gid != read_group
                or stat.S_IMODE(entry.st_mode) != 0o550):
            raise PeerIdentityError("Delegated identity directory must be read-only to its application group")
    for name in ("key.pem", "certificate.pem", "owner.pem"):
        entry = (directory / name).lstat()
        if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != owner_uid or entry.st_nlink != 1
                or entry.st_size > 65536):
            raise PeerIdentityError("Identity files must belong to the service owner")
        if read_group is not None and (entry.st_gid != read_group or stat.S_IMODE(entry.st_mode) != 0o440):
            raise PeerIdentityError("Delegated identity files must be read-only to the application group")
        if read_group is None and name == "key.pem" and stat.S_IMODE(entry.st_mode) & 0o077:
            raise PeerIdentityError("Device private key permissions are too broad")
    root = x509.load_pem_x509_certificate((directory / "owner.pem").read_bytes())
    if not isinstance(root.public_key(), Ed25519PublicKey) or key_id(public_bytes(root.public_key())) != policy.owner_id:
        raise PeerIdentityError("TLS owner key differs from the enrollment owner")
    own = x509.load_pem_x509_certificate((directory / "certificate.pem").read_bytes())
    if certificate_identity(own, policy) != identity:
        raise PeerIdentityError("Local certificate is for another enrolled device")
    result = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER if server else ssl.PROTOCOL_TLS_CLIENT)
    result.minimum_version = ssl.TLSVersion.TLSv1_3
    result.verify_mode = ssl.CERT_REQUIRED
    result.load_verify_locations(cadata=root.public_bytes(serialization.Encoding.PEM).decode())
    result.load_cert_chain(str(directory / "certificate.pem"), str(directory / "key.pem"))
    result.set_alpn_protocols([PROTOCOL])
    # Constructing the context directly avoids SSLKEYLOGFILE's ambient effect.
    result.keylog_filename = None
    return result


def authenticate(connection, policy, *, expected=None):
    if connection.selected_alpn_protocol() != PROTOCOL:
        raise PeerIdentityError("Peer protocol was not negotiated")
    der = connection.getpeercert(binary_form=True)
    if not der:
        raise PeerIdentityError("Peer certificate is required")
    identity = certificate_identity(x509.load_der_x509_certificate(der), policy)
    if expected is not None and identity != expected:
        raise PeerIdentityError("Connected device differs from the requested peer")
    return identity, hashlib.sha256(der).hexdigest()


class LabNetwork:
    """Explicit small private IPv4 subnets; never resolve or scan public names."""

    def __init__(self, cidrs=("127.0.0.0/24",)):
        if not 1 <= len(cidrs) <= 4:
            raise PeerIdentityError("Lab network list exceeds bounds")
        private = [ipaddress.ip_network(value) for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8")]
        self.networks = tuple(ipaddress.ip_network(value, strict=True) for value in cidrs)
        for network in self.networks:
            if network.version != 4 or network.prefixlen < 24 or not any(network.subnet_of(allowed) for allowed in private):
                raise PeerIdentityError("Only explicit private lab subnets of at most 256 addresses are permitted")

    def address(self, value):
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise PeerIdentityError("A literal enrolled-lab address is required; DNS is not used") from exc
        if address.version != 4 or address.is_multicast or address.is_unspecified or not any(address in net for net in self.networks):
            raise PeerIdentityError("Address is outside the declared lab")
        return str(address)
