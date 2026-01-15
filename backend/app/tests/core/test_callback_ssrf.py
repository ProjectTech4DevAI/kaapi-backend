import socket
from typing import Any
import requests
from unittest.mock import patch, MagicMock

import pytest

from app.utils import _is_private_ip, validate_callback_url, send_callback


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
        mock_response.raise_for_status.return_value = None
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
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
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
        mock_response.raise_for_status.return_value = None
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
        mock_response.raise_for_status.return_value = None
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
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b"test"]
        mock_session.post.return_value = mock_response
        mock_session_class.return_value.__enter__.return_value = mock_session

        test_data = {"status": "completed", "result": 42}
        send_callback("https://api.example.com/callback", test_data)

        call_kwargs = mock_session.post.call_args[1]
        assert "json" in call_kwargs
        assert call_kwargs["json"] == test_data
