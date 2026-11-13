FROM python:3.12-slim
WORKDIR /opt/worldline
COPY pyproject.toml README.md LICENSE ./
COPY worldline/ ./worldline/
RUN pip install --no-cache-dir '.[rua,cua]' && \
    PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.playwright.dev/dbazure/download/playwright \
    python -m playwright install --with-deps chromium && \
    apt-get update && apt-get install -y --no-install-recommends libegl1 libgl1-mesa-dri libosmesa6 && \
    rm -rf /var/lib/apt/lists/* && chmod -R a+rX /root/.cache/ms-playwright
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/browsers
RUN mv /root/.cache/ms-playwright /opt/browsers
CMD ["python", "-m", "worldline.runtime.worker"]
