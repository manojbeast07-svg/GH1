"""Basic application logging setup, shared by scripts and tests.

Kept deliberately minimal for Section 1: a single console handler with a
timestamped format. Later sections must avoid calling logger.debug/info
inside per-image or per-pixel loops, since that would distort benchmark
timings.
"""

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "xray_cuda_demo") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)

    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root = logging.getLogger("xray_cuda_demo")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        _CONFIGURED = True

    return logger
