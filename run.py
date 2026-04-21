"""Launch Streamlit app with ngrok tunnel or LAN-only HTTPS.

Usage:
    python run.py                          # Free ngrok (random URL each time)
    python run.py --token YOUR_AUTH_TOKEN  # Authenticated ngrok
    python run.py --lan                    # LAN-only HTTPS with self-signed cert
                                           #   (for mobile camera on same WiFi)

Mobile browsers require HTTPS for camera access (navigator.mediaDevices).
Plain http://<LAN-IP>:PORT will NOT work on a phone.
"""

import argparse
import datetime
import ipaddress
import socket
import subprocess
import sys
from pathlib import Path

STREAMLIT_PORT = 8501
CERT_DIR = Path(".certs")


def get_lan_ip() -> str:
    """Return the LAN IP used for the default outbound route."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def ensure_self_signed_cert(lan_ip: str) -> tuple[Path, Path]:
    """Generate a self-signed cert valid for localhost and the LAN IP.

    Regenerates if the cached cert's SAN does not include the current LAN IP.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    CERT_DIR.mkdir(exist_ok=True)
    cert_path = CERT_DIR / "streamlit.crt"
    key_path = CERT_DIR / "streamlit.key"

    # Reuse existing cert only if it still covers the current LAN IP.
    if cert_path.exists() and key_path.exists():
        try:
            existing = x509.load_pem_x509_certificate(cert_path.read_bytes())
            san = existing.extensions.get_extension_for_class(
                x509.SubjectAlternativeName).value
            ips = [str(i) for i in san.get_values_for_type(x509.IPAddress)]
            if lan_ip in ips:
                return cert_path, key_path
        except Exception:
            pass  # fall through and regenerate

    print(f"Generating self-signed certificate for {lan_ip}...")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "FAW Detector (LAN)"),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv4Address(lan_ip)),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    key_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def run_lan(port: int) -> None:
    """Run Streamlit with self-signed HTTPS on the LAN IP."""
    lan_ip = get_lan_ip()
    cert_path, key_path = ensure_self_signed_cert(lan_ip)

    url = f"https://{lan_ip}:{port}"
    print(f"\n{'=' * 60}")
    print(f"  LAN HTTPS URL: {url}")
    print(f"{'=' * 60}")
    print(f"\n  Open on your phone (same WiFi).")
    print(f"  Browser will warn about the self-signed cert —")
    print(f"  tap 'Advanced' → 'Proceed anyway'. This is expected.")
    print(f"\n  Press Ctrl+C to stop.\n")

    try:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run", "app.py",
                "--server.port", str(port),
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--server.sslCertFile", str(cert_path),
                "--server.sslKeyFile", str(key_path),
            ],
            check=True,
        )
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description="Run FAW Detector")
    parser.add_argument("--token", type=str, default=None,
                        help="ngrok auth token (get from https://dashboard.ngrok.com)")
    parser.add_argument("--port", type=int, default=STREAMLIT_PORT,
                        help="Streamlit port (default: 8501)")
    parser.add_argument("--lan", action="store_true",
                        help="LAN-only HTTPS with self-signed cert (skip ngrok). "
                             "Use when PC and phone are on the same WiFi.")
    args = parser.parse_args()

    if args.lan:
        run_lan(args.port)
        return

    # Configure ngrok auth token if provided
    from pyngrok import conf, ngrok

    if args.token:
        conf.get_default().auth_token = args.token
        print(f"ngrok auth token set.")

    # Start ngrok tunnel — force HTTPS so mobile browsers expose
    # navigator.mediaDevices (camera access is blocked on plain HTTP).
    print(f"\nStarting ngrok tunnel on port {args.port}...")
    tunnel = ngrok.connect(args.port, proto="http", bind_tls=True)
    public_url = tunnel.public_url
    if public_url.startswith("http://"):
        public_url = "https://" + public_url[len("http://"):]

    print(f"\n{'=' * 60}")
    print(f"  PUBLIC URL: {public_url}")
    print(f"{'=' * 60}")
    print(f"\n  Open this URL on your phone or any device!")
    print(f"  NOTE: Mobile camera access requires the HTTPS URL above.")
    print(f"        Plain http:// will fail with navigator.mediaDevices undefined.")
    print(f"  Local URL:  http://localhost:{args.port}")
    print(f"\n  Press Ctrl+C to stop.\n")

    # Start Streamlit
    try:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run", "app.py",
                "--server.port", str(args.port),
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down ngrok tunnel...")
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()


if __name__ == "__main__":
    main()
