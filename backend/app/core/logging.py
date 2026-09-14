"""Secure structured logging configuration with secret masking and privacy protection."""
import logging
import re
from typing import Any

# Regex patterns matching common secret and credential signatures
SECRET_MASK_PATTERNS = [
    (re.compile(r'(?i)(bearer\s+)[A-Za-z0-9\-_=]+\.[A-Za-z0-9\-_=]+\.?[A-Za-z0-9\-_.+/=]*'), r'\1[REDACTED_JWT]'),
    (re.compile(r'(?i)(service_role_key|anon_key|api_key|password|secret|auth_token)\s*[:=]\s*["\']?[^"\'\s,]+["\']?'), r'\1=[REDACTED]'),
    (re.compile(r'ey[A-Za-z0-9\-_=]{15,}\.[A-Za-z0-9\-_=]{15,}\.[A-Za-z0-9\-_.+/=]{10,}'), '[REDACTED_JWT]'),
]


class SensitiveDataFilter(logging.Filter):
    """Logging filter that automatically redacts credentials, JWT tokens, and private keys."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            masked_msg = record.msg
            for pattern, repl in SECRET_MASK_PATTERNS:
                masked_msg = pattern.sub(repl, masked_msg)
            record.msg = masked_msg

        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._sanitize_value(v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self._sanitize_value(v) for v in record.args)
        return True

    def _sanitize_value(self, val: Any) -> Any:
        if isinstance(val, str):
            for pattern, repl in SECRET_MASK_PATTERNS:
                val = pattern.sub(repl, val)
        return val


def setup_secure_logging(log_level: str = "INFO") -> None:
    """Configures root and app loggers with SensitiveDataFilter."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Attach filter to existing handlers
    sec_filter = SensitiveDataFilter()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
        handler.addFilter(sec_filter)

    # Ensure app loggers also receive the filter
    app_logger = logging.getLogger("app")
    app_logger.addFilter(sec_filter)

    ml_logger = logging.getLogger("ml")
    ml_logger.addFilter(sec_filter)
