import os, json, requests, time
from typing import List
from src.mcp_guard.schemas import DetectorResult, IssueType
from src.mcp_guard.exceptions import RemoteVerifyException
from src.mcp_guard.settings import settings

VERIFY_URL = os.getenv(
    "MCP_GUARD_REMOTE_VERIFY_URL",
    "https://mcp.example.com"
)

TIMEOUT = settings.VERIFY_TIMEOUT
CONF_THRESHOLD = float(os.getenv("MCP_GUARD_CONF_THRESHOLD", "0.5"))

class RemoteSignatureDetector:
    def detect(self, text: str) -> List[DetectorResult]:

        if VERIFY_URL.startswith("https://mcp.example.com"):
            return []

        payload = {"text": text}

        try:
            t0 = time.time()
            r = requests.post(VERIFY_URL, json=payload, timeout=TIMEOUT)

            if r.status_code != 200:
                raise RemoteVerifyException(
                    message=f"HTTP {r.status_code}: {r.text}",
                    url=VERIFY_URL,
                    status_code=r.status_code
                )

            print(f"[DEBUG] API raw response: {r.text}")
            try:
                data = r.json()
            except Exception as e:
                raise RemoteVerifyException(
                    message=f"API response content cannot be parsed as JSON: {r.text}",
                    url=VERIFY_URL
                )
            if not isinstance(data, dict):
                raise RemoteVerifyException(
                    message=f"API returned non-dictionary type: {data}",
                    url=VERIFY_URL
                )
            label = data.get("label")
            conf  = data.get("confidence", 0.0)

            if label == 1 and conf >= CONF_THRESHOLD:
                return [
                    DetectorResult(
                        type=IssueType.SIGNATURE,
                        message=f"Model detected an issue, confidence {conf:.2%}",
                        location="description"
                    )
                ]

            return []

        except requests.RequestException as e:
            raise RemoteVerifyException(
                message=f"Request failed: {e}",
                url=VERIFY_URL
            )

        except Exception as e:
            raise RemoteVerifyException(
                message=f"Unknown exception: {e}",
                url=VERIFY_URL
            )