FROM python:3.11-alpine

# Install runtime dependencies needed by tools (bash, curl, etc.)
RUN apk add --no-cache bash curl tree

RUN adduser -D -u 1000 -h /home/sandbox sandbox && \
    mkdir -p /workspace && chown sandbox:sandbox /workspace

WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY tools/ ./tools/
COPY mcp_server.py .

# Drop to non-root user
USER sandbox

# Workspace is the default working directory for tool execution
ENV SANDBOX_WORKSPACE=/workspace

ENV PYTHONUNBUFFERED=1

# Keep container alive; MCP server spawned on-demand via docker exec
ENTRYPOINT ["sleep", "infinity"]
