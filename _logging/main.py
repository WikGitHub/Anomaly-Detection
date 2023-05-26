import logging
import os
import sys


def get_logger(name: str = None, propergate: bool = False) -> logging.Logger:

    logger = logging.getLogger(name=name)
    logger.handlers.clear()
    ch = logging.StreamHandler(stream=sys.stdout)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)3d-%(name)-12s: %(levelname)-8s %(message)s"
    )
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    loglevel = None
    split_name = name.split(".")
    for i in range(len(split_name), 0, -1):
        env_name = ".".join([n.upper() for n in split_name[:i]])
        loglevel = os.getenv(key=f"{env_name}_LOGLEVEL")
        if loglevel is not None:
            break

    if loglevel is None:
        loglevel = os.getenv(key="LOGLEVEL", default="INFO")

    loglevel = getattr(logging, loglevel.upper())
    logger.setLevel(level=loglevel)
    logger.propagate = propergate

    return logger
