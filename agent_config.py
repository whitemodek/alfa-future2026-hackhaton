"""Validated resource limits; secrets never belong in configuration files."""
from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class Config:
    total_seconds: float = 10500
    reserve_seconds: float = 900
    call_seconds: float = 1800
    api_seconds: float = 90
    max_steps: int = 60
    max_api_attempts: int = 80
    max_tokens: int = 200000
    response_tokens: int = 4096
    context_bytes: int = 100000
    retries: int = 2
    memory_mb: int = 28000
    threads: int = 8

    def __post_init__(self):
        integer_fields = ('max_steps', 'max_api_attempts', 'max_tokens', 'response_tokens',
                          'context_bytes', 'retries', 'memory_mb', 'threads')
        if any(type(getattr(self, key)) is not int for key in integer_fields):
            raise ValueError('counts and resource sizes must be integers')
        for key, value in vars(self).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            if value < 0 or (key not in ('reserve_seconds', 'retries') and value == 0):
                raise ValueError(f"invalid {key}")
        if self.reserve_seconds >= self.total_seconds or self.total_seconds > 10800:
            raise ValueError('reserve must be below total; total must be <= 10800 seconds')
        if self.threads > 8 or self.memory_mb > 30000:
            raise ValueError('resource limits exceed platform allowance')

    @classmethod
    def from_env(cls):
        defaults = cls()
        return cls(**{key: (float if key.endswith('_seconds') else int)(os.environ.get('AGENT_' + key.upper(), value))
                      for key, value in vars(defaults).items()})
