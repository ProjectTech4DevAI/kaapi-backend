import itertools
import json
import socket
from typing import Any
import requests
from unittest.mock import patch, MagicMock

import pytest

import hashlib
import hmac

from app.utils import (
    _is_private_ip,
    validate_callback_url,
    send_callback,
    sign_webhook_payload,
)

pytestmark = pytest.mark.usefixtures("no_retry_backoff")


class TestIsPrivateIP:
    """Test suite for _is_private_ip function."""

    def test_private_ipv4_addresses(self) -> None:
        """Test that private IPv4 addresses are correctly identified."""
        private_ips = [
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.0.1",
            "192.168.255.255",
        ]
        for ip in private_ips:
            is_blocked, reason = _is_private_ip(ip)
            assert is_blocked is True, f"{ip} should be identified as private"
            assert reason == "private", f"{ip} should have reason 'private'"

    def test_localhost_addresses(self) -> None:
        """Test that localhost/loopback addresses are blocked."""
        localhost_ips = [
            "127.0.0.1",
            "127.0.0.2",
            "127.255.255.255",
            "::1",
        ]
        for ip in localhost_ips:
            is_blocked, reason = _is_private_ip(ip)
            assert is_blocked is True, f"{ip} should be identified as loopback"
            assert (
                reason == "loopback/localhost"
            ), f"{ip} should have reason 'loopback/localhost'"

    def test_link_local_addresses(self) -> None:
        """Test that link-local addresses are blocked."""
        link_local_ips = [
            "169.254.0.1",
            "169.254.169.254",
            "169.254.255.255",
        ]
        for ip in link_local_ips:
            is_blocked, reason = _is_private_ip(ip)
            assert is_blocked is True, f"{ip} should be identified as link-local"
            assert reason == "link-local", f"{ip} should have reason 'link-local'"

    def test_multicast_addresses(self) -> None:
        """Test that multicast addresses are blocked."""
        multicast_ips = [
            "224.0.0.1",
            "239.255.255.255",
        ]
        for ip in multicast_ips:
            is_blocked, reason = _is_private_ip(ip)
            assert is_blocked is True, f"{ip} should be identified as multicast"
            assert reason == "multicast", f"{ip} should have reason 'multicast'"

    def test_public_ipv4_addresses(self) -> None:
        """Test that public IPv4 addresses are not blocked."""
        public_ips = [
            "8.8.8.8",
            "1.1.1.1",
            "93.184.216.34",
            "151.101.1.140",
        ]
        for ip in public_ips:
            is_blocked, reason = _is_private_ip(ip)
            assert is_blocked is False, f"{ip} should be identified as public"
            assert reason == "", f"{ip} should have empty reason"

    def test_public_ipv6_addresses(self) -> None:
        """Test that public IPv6 addresses are not blocked."""
        public_ipv6 = [
            "2001:4860:4860::8888",
            "2606:4700:4700::1111",
        ]
        for ip in public_ipv6:
            is_blocked, reason = _is_private_ip(ip)
            assert is_blocked is False, f"{ip} should be identified as public"
            assert reason == "", f"{ip} should have empty reason"

    def test_invalid_ip_addresses(self) -> None:
        """Test that invalid IP addresses return False."""
        invalid_ips = [
            "not_an_ip",
            "999.999.999.999",
            "example.com",
        ]
        for ip in invalid_ips:
            is_blocked, reason = _is_private_ip(ip)
            assert is_blocked is False, f"{ip} should return False"
            assert reason == "", f"{ip} should have empty reason"


class TestValidateCallbackURL:
    """Test suite for validate_callback_url function."""

    def test_reject_non_https_schemes(self) -> None:
        """Test that non-HTTPS URL schemes are rejected."""
        non_https_urls = [
            "http://example.com/callback",
            "ftp://example.com/callback",
            "file:///etc/passwd",
        ]
        for url in non_https_urls:
            with pytest.raises(ValueError, match="Only HTTPS URLs are allowed"):
                validate_callback_url(url)

    @patch("socket.getaddrinfo")
    def test_reject_localhost_by_name(self, mock_getaddrinfo: Any) -> None:
        """Test that localhost is rejected."""
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))
        ]

        with pytest.raises(ValueError, match="loopback/localhost IP address"):
            validate_callback_url("https://localhost/callback")

    @patch("socket.getaddrinfo")
    def test_reject_private_ip_addresses(self, mock_getaddrinfo: Any) -> None:
        """Test that private IPs in all RFC 1918 ranges are rejected."""
        private_ips = [
            ("10.0.0.1", "https://internal.company.com/callback"),
            ("192.168.1.1", "https://router.local/callback"),
            ("172.16.0.1", "https://internal-api.local/callback"),
        ]

        for ip, url in private_ips:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))
            ]

            with pytest.raises(ValueError, match="private IP address"):
                validate_callback_url(url)

    @patch("socket.getaddrinfo")
    def test_reject_link_local_addresses(self, mock_getaddrinfo: Any) -> None:
        """Test that link-local addresses are rejected (including cloud metadata endpoints)."""
        link_local_ips = [
            (
                "169.254.169.254",
                "https://metadata.aws/callback",
            ),  # AWS metadata endpoint
            ("169.254.0.1", "https://link-local.example/callback"),
        ]

        for ip, url in link_local_ips:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))
            ]

            with pytest.raises(ValueError, match="link-local IP address"):
                validate_callback_url(url)

    @patch("socket.getaddrinfo")
    def test_accept_public_ip_addresses(self, mock_getaddrinfo: Any) -> None:
        """Test that valid HTTPS URLs with public IP addresses are accepted."""
        public_ips = [
            ("8.8.8.8", "https://api.example.com/callback"),
            ("151.101.1.140", "https://webhook.site/unique-id"),
        ]

        for ip, url in public_ips:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))
            ]

            validate_callback_url(url)

    def test_reject_url_without_hostname(self) -> None:
        """Test that URLs without hostname are rejected."""
        with pytest.raises(ValueError, match="URL must have a valid hostname"):
            validate_callback_url("https:///callback")

    def test_reject_invalid_url_format(self) -> None:
        """Test that invalid URL formats are rejected."""
        with pytest.raises(ValueError, match="Only HTTPS URLs are allowed"):
            validate_callback_url("not a url at all")

    @patch("socket.getaddrinfo")
    def test_check_all_resolved_ips(self, mock_getaddrinfo: Any) -> None:
        """Test that all resolved IPs are checked (DNS round-robin)."""
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.1", 443)),
        ]

        with pytest.raises(ValueError, match="private IP address"):
            validate_callback_url("https://malicious-dns.example/callback")

    @patch("socket.getaddrinfo")
    def test_ipv6_public_address_accepted(self, mock_getaddrinfo: Any) -> None:
        """Test that public IPv6 addresses are accepted."""
        mock_getaddrinfo.return_value = [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:4860:4860::8888", 443))
        ]

        validate_callback_url("https://ipv6.example.com/callback")

    @patch("socket.getaddrinfo")
    def test_ipv6_localhost_rejected(self, mock_getaddrinfo: Any) -> None:
        """Test that IPv6 localhost is rejected."""
        mock_getaddrinfo.return_value = [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 443))
        ]

        with pytest.raises(ValueError, match="loopback/localhost IP address"):
            validate_callback_url("https://localhost6/callback")


class TestSendCallback:
    """Test suite for send_callback function."""

    @patch("app.utils.validate_callback_url")
    @patch("requests.Session")
    def test_successful_callback(
        self, mock_session_class: Any, mock_validate: Any
    ) -> None:
        """Test successful callback execution."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"test"]
        mock_session.post.return_value = mock_response
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = send_callback(
            "https://api.example.com/callback", {"status": "success"}
        )

        assert result is True
        mock_session.post.assert_called_once()
        assert mock_session.post.call_args[1]["allow_redirects"] is False

    @patch("app.utils.validate_callback_url")
    @patch("requests.Session")
    def test_callback_network_error(
        self, mock_session_class: Any, mock_validate: Any
    ) -> None:
        """Test that callback returns False on network errors."""
        mock_session = MagicMock()
        mock_session.post.side_effect = requests.RequestException("Connection refused")
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = send_callback("https://api.example.com/callback", {"data": "test"})

        assert result is False

    @patch("app.utils.validate_callback_url")
    @patch("requests.Session")
    def test_callback_http_error(
        self, mock_session_class: Any, mock_validate: Any
    ) -> None:
        """Test that callback returns False on HTTP errors."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.post.return_value = mock_response
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = send_callback("https://api.example.com/callback", {"data": "test"})

        assert result is False

    @patch("app.utils.validate_callback_url")
    @patch("requests.Session")
    def test_callback_disables_redirects(
        self, mock_session_class: Any, mock_validate: Any
    ) -> None:
        """Test that redirects are disabled to prevent redirect-based SSRF."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"test"]
        mock_session.post.return_value = mock_response
        mock_session_class.return_value.__enter__.return_value = mock_session

        send_callback("https://api.example.com/callback", {"data": "test"})

        call_kwargs = mock_session.post.call_args[1]
        assert call_kwargs["allow_redirects"] is False

    @patch("app.utils.validate_callback_url")
    @patch("requests.Session")
    def test_callback_uses_timeout(
        self, mock_session_class: Any, mock_validate: Any
    ) -> None:
        """Test that callback uses configured timeouts."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"test"]
        mock_session.post.return_value = mock_response
        mock_session_class.return_value.__enter__.return_value = mock_session

        send_callback("https://api.example.com/callback", {"data": "test"})

        call_kwargs = mock_session.post.call_args[1]
        assert "timeout" in call_kwargs
        assert isinstance(call_kwargs["timeout"], tuple)
        assert len(call_kwargs["timeout"]) == 2

    @patch("app.utils.validate_callback_url")
    @patch("requests.Session")
    def test_callback_sends_json_data(
        self, mock_session_class: Any, mock_validate: Any
    ) -> None:
        """Test that callback sends data as JSON."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"test"]
        mock_session.post.return_value = mock_response
        mock_session_class.return_value.__enter__.return_value = mock_session

        test_data = {"status": "completed", "result": 42}
        send_callback("https://api.example.com/callback", test_data)

        call_kwargs = mock_session.post.call_args[1]
        # Body is now pre-serialized so HMAC signs the exact bytes we send.
        assert "data" in call_kwargs
        assert (
            call_kwargs["data"] == json.dumps(test_data, separators=(",", ":")).encode()
        )
        assert call_kwargs["headers"]["Content-Type"] == "application/json"
        # No webhook_secret passed → no signature headers.
        assert "X-Webhook-Signature" not in call_kwargs["headers"]
        assert "X-Webhook-Timestamp" not in call_kwargs["headers"]


class TestSignWebhookPayload:
    def test_returns_hex_signature_and_timestamp(self):
        sig, ts = sign_webhook_payload("secret", b"body")
        assert isinstance(sig, str) and len(sig) == 64  # sha256 hex
        assert isinstance(ts, int) and ts > 0

    def test_deterministic_with_fixed_timestamp(self):
        body = b'{"key":"value"}'
        sig1, _ = sign_webhook_payload("secret", body, timestamp_ms=1000)
        sig2, _ = sign_webhook_payload("secret", body, timestamp_ms=1000)
        assert sig1 == sig2

    def test_different_secrets_produce_different_signatures(self):
        body = b"payload"
        sig1, _ = sign_webhook_payload("secret-a", body, timestamp_ms=1000)
        sig2, _ = sign_webhook_payload("secret-b", body, timestamp_ms=1000)
        assert sig1 != sig2

    def test_different_timestamps_produce_different_signatures(self):
        body = b"payload"
        sig1, _ = sign_webhook_payload("secret", body, timestamp_ms=1000)
        sig2, _ = sign_webhook_payload("secret", body, timestamp_ms=2000)
        assert sig1 != sig2

    def test_signature_matches_manual_hmac(self):
        secret, body, ts = "mysecret", b"hello", 999
        expected = hmac.new(
            secret.encode(), f"{ts}.".encode() + body, hashlib.sha256
        ).hexdigest()
        sig, returned_ts = sign_webhook_payload(secret, body, timestamp_ms=ts)
        assert sig == expected
        assert returned_ts == ts


def _callback_response(status_code: int) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    return response


@patch("app.utils.validate_callback_url")
class TestSendCallbackRetry:
    URL = "https://api.example.com/callback"

    def test_recovers_after_transient_failures(self, _validate: Any) -> None:
        # 302 counts as a failure: redirects are off, so nothing was delivered.
        outcomes = [
            _callback_response(302),
            requests.ConnectionError("reset"),
            _callback_response(500),
            _callback_response(200),
        ]
        with patch.object(requests.Session, "post", side_effect=outcomes) as post:
            assert send_callback(self.URL, {"status": "done"}) is True
        assert post.call_count == 4

    @pytest.mark.parametrize("status_code", [503, 404])
    def test_gives_up_after_four_attempts(
        self, _validate: Any, status_code: int
    ) -> None:
        with patch.object(
            requests.Session, "post", return_value=_callback_response(status_code)
        ) as post:
            assert send_callback(self.URL, {"status": "done"}) is False
        assert post.call_count == 4

    def test_each_attempt_is_signed_with_a_fresh_timestamp(
        self, _validate: Any
    ) -> None:
        with (
            patch("app.utils.time.time", side_effect=itertools.count(1_700_000_000)),
            patch.object(
                requests.Session, "post", return_value=_callback_response(503)
            ) as post,
        ):
            send_callback(self.URL, {"status": "done"}, webhook_secret="s3cret")

        timestamps = {
            c.kwargs["headers"]["X-Webhook-Timestamp"] for c in post.mock_calls
        }
        assert len(timestamps) == 4

    def test_invalid_url_is_never_posted(self, validate: Any) -> None:
        validate.side_effect = ValueError("private IP")
        with patch.object(requests.Session, "post") as post:
            assert send_callback(self.URL, {"status": "done"}) is False
        post.assert_not_called()
