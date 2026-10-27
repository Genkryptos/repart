# RePart web app. OpenSCAD comes from the official image (dated dev build, has Manifold).
#   docker build -t repart .
#   docker run -p 8000:8000 -e MODEL_API_KEY=... -e REPART_ACCESS_CODE=... repart
FROM openscad/openscad:dev.2026-10-20

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python \
    PYTHONUNBUFFERED=1 REPART_TRUST_PROXY=1

WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY data/parts ./data/parts
RUN uv sync --frozen --no-dev

EXPOSE 8000
ENTRYPOINT []
CMD ["uv", "run", "--no-sync", "repart", "serve", "--host", "0.0.0.0", "--port", "8000"]
