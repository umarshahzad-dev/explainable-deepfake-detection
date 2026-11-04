"""Runtime URL ingestion only; import and all unit tests perform no network calls."""
import http.client
import ipaddress
import socket
import ssl
import time
from urllib.parse import urlsplit, urljoin, quote
from deepfake_detection.inference.engine import MAX_BYTES


def validate_url(url, *, resolver=socket.getaddrinfo):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only public HTTPS image URLs without credentials are allowed")
    if parsed.port not in {None, 443}:
        raise ValueError("Only HTTPS port 443 is allowed")
    host = parsed.hostname.encode("idna").decode("ascii")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None and (not literal.is_global or literal.is_multicast):
        raise ValueError("Private, loopback, link-local and reserved destinations are forbidden")
    answers = resolver(host, 443)
    addresses = sorted({entry[4][0] for entry in answers})
    if not addresses or any(not ipaddress.ip_address(ip).is_global or ipaddress.ip_address(ip).is_multicast for ip in addresses):
        raise ValueError("DNS must resolve exclusively to public addresses")
    return parsed, host, addresses[0]


class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, hostname, address, *, timeout):
        super().__init__(hostname, 443, timeout=timeout, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        # Pin the validated destination IP while preserving TLS hostname verification.
        raw = socket.create_connection((self.address, self.port), self.timeout)
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except Exception:
            raw.close()
            raise


def fetch_public_image(url, *, max_bytes=MAX_BYTES, timeout=20, max_redirects=3):
    deadline = time.monotonic() + timeout
    for hop in range(max_redirects + 1):
        parsed, hostname, address = validate_url(url)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Image fetch deadline exceeded")
        connection = PinnedHTTPS(hostname, address, timeout=min(10, remaining))
        try:
            path = quote(parsed.path or "/", safe="/%:@")
            if parsed.query:
                path += "?" + quote(parsed.query, safe="%=&;:+,/?@")
            connection.request("GET", path, headers={"User-Agent": "GraduationDeepfakeExperiment/1.0", "Accept": "image/png,image/jpeg"})
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location or hop == max_redirects:
                    raise ValueError("Missing or excessive image redirects")
                url = urljoin(url, location)
                continue  # Every redirect is revalidated, including its DNS.
            if response.status != 200:
                raise ValueError(f"Image server returned HTTP {response.status}")
            if not response.getheader("Content-Type", "").split(";")[0].lower().startswith("image/"):
                raise ValueError("URL did not return image content")
            length = response.getheader("Content-Length")
            if length and int(length) > max_bytes:
                raise ValueError("Remote image exceeds byte budget")
            chunks, size = [], 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Image fetch deadline exceeded")
                if connection.sock is not None:
                    connection.sock.settimeout(min(10, remaining))
                chunk = response.read(min(65536, max_bytes + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError("Remote image exceeds byte budget")
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            connection.close()
    raise ValueError("Too many redirects")
