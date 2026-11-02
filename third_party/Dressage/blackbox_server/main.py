from __future__ import annotations

import logging

import uvicorn

from blackbox_server.app import create_app
from blackbox_server.config import BlackboxServerConfig


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = BlackboxServerConfig.from_env()
    uvicorn.run(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
