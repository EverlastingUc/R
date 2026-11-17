"""Loopback-only HTTP API for the Valence conversational engine."""

from __future__ import annotations

import ipaddress
import json
import secrets
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, RLock
from urllib.parse import urlsplit

from proton_i.valence.backend import ChatBackend
from proton_i.valence.engine import ValenceEngine, ValenceError
from proton_i.valence.tools import CapabilityRegistry, ExecutionPolicy

DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8765
MAX_REQUEST_BYTES = 16 * 1024
MAX_SESSIONS = 64


@dataclass
class _Conversation:
    """One isolated in-memory conversation and its request lock."""

    engine: ValenceEngine
    lock: Lock


class ValenceHTTPServer(ThreadingHTTPServer):
    """Threaded local server with bounded in-memory Valence sessions."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        backend: ChatBackend,
        max_sessions: int,
        registry: CapabilityRegistry | None = None,
        tool_policy: ExecutionPolicy | None = None,
    ) -> None:
        """Create a loopback server and its isolated conversation registry."""
        self.backend = backend
        self.max_sessions = max_sessions
        self.registry = registry
        self.tool_policy = tool_policy or ExecutionPolicy()
        self.sessions: dict[str, _Conversation] = {}
        self.sessions_lock = RLock()
        super().__init__(server_address, ValenceRequestHandler)

    def create_session(self) -> tuple[str, _Conversation] | None:
        """Create a session unless the configured in-memory limit is reached."""
        with self.sessions_lock:
            if len(self.sessions) >= self.max_sessions:
                return None
            session_id = secrets.token_urlsafe(24)
            session = _Conversation(
                engine=ValenceEngine(
                    self.backend,
                    registry=self.registry,
                    tool_policy=self.tool_policy,
                ),
                lock=Lock(),
            )
            self.sessions[session_id] = session
            return session_id, session

    def get_session(self, session_id: str) -> _Conversation | None:
        """Return an existing session without creating a new one."""
        with self.sessions_lock:
            return self.sessions.get(session_id)

    @property
    def active_sessions(self) -> int:
        """Return the current number of in-memory sessions."""
        with self.sessions_lock:
            return len(self.sessions)


class ValenceRequestHandler(BaseHTTPRequestHandler):
    """JSON request handler with local-origin and payload-size checks."""

    server: ValenceHTTPServer
    server_version = "ValenceAPI/0.1"
    sys_version = ""

    def log_message(self, _format: str, *_args: object) -> None:
        """Avoid writing prompts, session identifiers, or client data to logs."""

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        """Send a JSON response with browser-safe and no-store headers."""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _origin_is_local(self) -> bool:
        """Reject browser requests whose Origin points to a non-loopback host."""
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        try:
            parsed = urlsplit(origin)
            origin_port = parsed.port or 80
        except ValueError:
            return False
        if parsed.scheme != "http" or not parsed.hostname:
            return False
        if parsed.hostname.lower() not in {"127.0.0.1", "localhost"}:
            return False
        return origin_port == self.server.server_port

    def _read_json(self) -> dict[str, object] | None:
        """Read one bounded JSON object or send a structured client error."""
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type.lower() != "application/json":
            self._send_json(415, {"error": "Content-Type must be application/json."})
            return None

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._send_json(411, {"error": "Content-Length is required."})
            return None
        if not raw_length.isdecimal():
            self._send_json(400, {"error": "Content-Length must be a valid integer."})
            return None

        body_length = int(raw_length)
        if body_length < 1:
            self._send_json(400, {"error": "Request body must not be empty."})
            return None
        if body_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "Request body exceeds the 16 KB limit."})
            return None

        body = self.rfile.read(body_length)
        if len(body) != body_length:
            self._send_json(400, {"error": "Request body was incomplete."})
            return None
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(
                400, {"error": "Request body must contain valid UTF-8 JSON."}
            )
            return None
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "JSON request body must be an object."})
            return None
        return payload

    def do_GET(self) -> None:
        """Expose liveness and the explicit capability matrix."""
        if not self._origin_is_local():
            self._send_json(
                403, {"error": "Only same-origin loopback requests are allowed."}
            )
            return
        path = urlsplit(self.path).path
        if path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "valence",
                    "active_sessions": self.server.active_sessions,
                },
            )
            return
        if path == "/v1/capabilities":
            self._send_json(
                200,
                {
                    "local_chat": "supported",
                    "in_memory_conversations": "supported",
                    "model_tool_calling": (
                        "supported"
                        if callable(getattr(self.server.backend, "complete_turn", None))
                        else "text_only"
                    ),
                    "web_search": (
                        "enabled"
                        if "search_web" in self.server.tool_policy.allowed_tools
                        else (
                            "disabled_by_policy"
                            if self.server.registry is not None
                            else "not_enabled"
                        )
                    ),
                    "mathematics_engine": "not_integrated",
                    "file_understanding": "not_integrated",
                    "artifact_creation": "not_integrated",
                    "model_comparison": "not_integrated",
                },
            )
            return
        self._send_json(404, {"error": "Endpoint not found."})

    def do_POST(self) -> None:
        """Handle chat and conversation-reset requests."""
        if not self._origin_is_local():
            self._send_json(
                403, {"error": "Only same-origin loopback browser origins are allowed."}
            )
            return

        path = urlsplit(self.path).path
        if path not in {"/v1/chat", "/v1/reset"}:
            self._send_json(404, {"error": "Endpoint not found."})
            return

        payload = self._read_json()
        if payload is None:
            return

        session_value = payload.get("session_id")
        if not isinstance(session_value, str) and session_value is not None:
            self._send_json(400, {"error": "session_id must be a string."})
            return
        if isinstance(session_value, str) and (
            not session_value or len(session_value) > 128
        ):
            self._send_json(
                400, {"error": "session_id must contain 1 to 128 characters."}
            )
            return

        if path == "/v1/reset":
            self._reset_session(session_value)
            return
        self._chat(payload, session_value)

    def _chat(self, payload: dict[str, object], session_id: str | None) -> None:
        """Generate one reply while serializing turns within the same session."""
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            self._send_json(400, {"error": "message must be a non-empty string."})
            return
        if len(message) > 12_000:
            self._send_json(
                413, {"error": "message exceeds the 12,000 character limit."}
            )
            return

        if session_id is None:
            created = self.server.create_session()
            if created is None:
                self._send_json(
                    429, {"error": "The in-memory session limit has been reached."}
                )
                return
            session_id, session = created
        else:
            session = self.server.get_session(session_id)
            if session is None:
                self._send_json(404, {"error": "Conversation session was not found."})
                return

        with session.lock:
            try:
                answer = session.engine.ask(message)
            except ValenceError as error:
                self._send_json(502, {"error": str(error), "session_id": session_id})
                return

        self._send_json(200, {"session_id": session_id, "message": answer})

    def _reset_session(self, session_id: str | None) -> None:
        """Clear a named conversation without deleting its session identifier."""
        if session_id is None:
            self._send_json(400, {"error": "session_id is required for reset."})
            return
        session = self.server.get_session(session_id)
        if session is None:
            self._send_json(404, {"error": "Conversation session was not found."})
            return
        with session.lock:
            session.engine.reset()
        self._send_json(200, {"session_id": session_id, "status": "reset"})


def create_server(
    backend: ChatBackend,
    host: str = DEFAULT_API_HOST,
    port: int = DEFAULT_API_PORT,
    max_sessions: int = MAX_SESSIONS,
    registry: CapabilityRegistry | None = None,
    tool_policy: ExecutionPolicy | None = None,
) -> ValenceHTTPServer:
    """Create a Valence API server bound exclusively to an IPv4 loopback address."""
    if not isinstance(host, str):
        raise TypeError("host must be a loopback IPv4 address")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("Valence API must bind to an IPv4 loopback address") from error
    if address.version != 4 or not address.is_loopback:
        raise ValueError("Valence API must bind to an IPv4 loopback address")
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    if (
        not isinstance(max_sessions, int)
        or isinstance(max_sessions, bool)
        or max_sessions < 1
    ):
        raise ValueError("max_sessions must be a positive integer")
    if not callable(getattr(backend, "complete", None)):
        raise TypeError("backend must provide a complete(messages) method")
    if registry is not None and not isinstance(registry, CapabilityRegistry):
        raise TypeError("registry must be a CapabilityRegistry or None")
    if tool_policy is not None and not isinstance(tool_policy, ExecutionPolicy):
        raise TypeError("tool_policy must be an ExecutionPolicy or None")
    if registry is None and tool_policy is not None and tool_policy.allowed_tools:
        raise ValueError("tool_policy cannot allow tools without a registry")
    return ValenceHTTPServer(
        (host, port),
        backend,
        max_sessions,
        registry=registry,
        tool_policy=tool_policy,
    )


def main() -> int:
    """Run the local API server with the configured Ollama model."""
    import argparse
    import os

    from proton_i.valence.backend import DEFAULT_MODEL, OllamaChatBackend
    from proton_i.valence.integrations.gluons import register_gluons_tool
    from proton_i.valence.tools import CapabilityRegistry, ExecutionPolicy

    parser = argparse.ArgumentParser(
        prog="valence-api",
        description="Run the loopback-only Valence HTTP API.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("VALENCE_MODEL", DEFAULT_MODEL),
        help="installed Ollama model name",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument(
        "--enable-web-search",
        action="store_true",
        help="allow the model to call Gluons website search",
    )
    arguments = parser.parse_args()

    backend = OllamaChatBackend(model=arguments.model)
    registry = None
    tool_policy = None
    if arguments.enable_web_search:
        registry = CapabilityRegistry()
        register_gluons_tool(registry)
        tool_policy = ExecutionPolicy(frozenset({"search_web"}))
    server = create_server(
        backend,
        port=arguments.port,
        registry=registry,
        tool_policy=tool_policy,
    )
    print(f"Valence API listening on http://{DEFAULT_API_HOST}:{arguments.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nValence API stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
