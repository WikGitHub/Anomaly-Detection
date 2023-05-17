import logging
import os
import sys


def get_logger(name: str = None, propergate: bool = False) -> logging.Logger:
    # get logger
    logger = logging.getLogger(name=name)

    # add handlers and formatting
    logger.handlers.clear()
    ch = logging.StreamHandler(stream=sys.stdout)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)3d-%(name)-12s: %(levelname)-8s %(message)s"
    )
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    # set level on module/package wise basis
    # starting with most specific name i.e. PACKAGE.PACKAGE.MODULE, then PACKAGE.PACKAGE, then PACKAGE...
    loglevel = None
    split_name = name.split(".")
    for i in range(len(split_name), 0, -1):
        env_name = ".".join([n.upper() for n in split_name[:i]])
        loglevel = os.getenv(key=f"{env_name}_LOGLEVEL")
        if loglevel is not None:
            break

    # otherwise use the global log level
    if loglevel is None:
        loglevel = os.getenv(key="LOGLEVEL", default="INFO")

    loglevel = getattr(logging, loglevel.upper())
    logger.setLevel(level=loglevel)
    logger.propagate = propergate

    return logger
