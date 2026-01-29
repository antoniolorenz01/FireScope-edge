from __future__ import annotations

import uvicorn

from app.config import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run("app.main:app", host=settings.app.host, port=settings.app.port, reload=False)


if __name__ == "__main__":
    main()
