#!/usr/bin/env python3
"""
Cron script to invoke an API endpoint periodically.
Uses async HTTP client to be resource-efficient.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Configuration
ENDPOINT = "/api/v1/cron/evaluations"  # Endpoint to invoke
REQUEST_TIMEOUT = 30  # Timeout for requests in seconds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EndpointInvoker:
    """Handles periodic endpoint invocation with authentication."""

    def __init__(self):
        # Load BASE_URL from environment with default fallback
        base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.base_url = base_url.rstrip("/")
        self.endpoint = ENDPOINT

        # Load interval from environment with default of 5 minutes
        self.interval_minutes = int(os.getenv("CRON_INTERVAL_MINUTES", "5"))
        self.interval_seconds = self.interval_minutes * 60
        self.access_token = None
        self.token_expiry = None

        # Load credentials from environment
        self.email = os.getenv("FIRST_SUPERUSER")
        self.password = os.getenv("FIRST_SUPERUSER_PASSWORD")

        if not self.email or not self.password:
            raise ValueError(
                "FIRST_SUPERUSER and FIRST_SUPERUSER_PASSWORD must be set in environment"
            )

    async def authenticate(self, client: httpx.AsyncClient) -> str:
        """Authenticate and get access token."""
        logger.info("Authenticating with API...")

        login_data = {
            "username": self.email,
            "password": self.password,
        }

        try:
            response = await client.post(
                f"{self.base_url}/api/v1/login/access-token",
                data=login_data,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            data = response.json()
            self.access_token = data.get("access_token")

            if not self.access_token:
                raise ValueError("No access token in response")

            logger.info("Authentication successful")
            return self.access_token

        except httpx.HTTPStatusError as e:
            logger.error(f"Authentication failed with status {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise

    async def invoke_endpoint(self, client: httpx.AsyncClient) -> dict:
        """Invoke the configured endpoint."""
        if not self.access_token:
            await self.authenticate(client)

        headers = {"Authorization": f"Bearer {self.access_token}"}

        # Debug: Log what we're sending
        logger.debug(f"Request URL: {self.base_url}{self.endpoint}")
        logger.debug(f"Request headers: {headers}")

        try:
            response = await client.get(
                f"{self.base_url}{self.endpoint}",
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

            # Debug: Log response headers and first part of body
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")

            # If unauthorized or forbidden (token expired/invalid), re-authenticate and retry once
            if response.status_code in (401, 403):
                logger.info("Token expired or invalid, re-authenticating...")
                await self.authenticate(client)
                headers = {"Authorization": f"Bearer {self.access_token}"}
                response = await client.get(
                    f"{self.base_url}{self.endpoint}",
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Endpoint invocation failed with status {e.response.status_code}: {e.response.text}"
            )
            raise
        except Exception as e:
            logger.error(f"Endpoint invocation error: {e}")
            raise

    async def run(self):
        """Main loop to invoke endpoint periodically."""
        logger.info(f"Using API Base URL: {self.base_url}")
        logger.info(
            f"Starting cron job - invoking {self.endpoint} every {self.interval_minutes} minutes"
        )

        # Use async context manager to ensure proper cleanup
        async with httpx.AsyncClient() as client:
            # Authenticate once at startup
            await self.authenticate(client)

            while True:
                try:
                    start_time = datetime.now()
                    logger.info(f"Invoking endpoint at {start_time}")

                    result = await self.invoke_endpoint(client)
                    logger.info(f"Endpoint invoked successfully: {result}")

                    # Calculate next invocation time
                    elapsed = (datetime.now() - start_time).total_seconds()
                    sleep_time = max(0, self.interval_seconds - elapsed)

                    if sleep_time > 0:
                        logger.info(
                            f"Sleeping for {sleep_time:.1f} seconds until next invocation"
                        )
                        await asyncio.sleep(sleep_time)

                except KeyboardInterrupt:
                    logger.info("Shutting down gracefully...")
                    break
                except Exception as e:
                    logger.error(f"Error during invocation: {e}")
                    # Wait before retrying on error
                    logger.info(f"Waiting {self.interval_seconds} seconds before retry")
                    await asyncio.sleep(self.interval_seconds)


def main():
    """Entry point for the script."""
    # Load environment variables
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        logger.warning(f"No .env file found at {env_path}")

    try:
        invoker = EndpointInvoker()
        asyncio.run(invoker.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
