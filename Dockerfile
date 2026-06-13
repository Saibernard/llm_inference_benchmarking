# Harness image (the load generator + reporter). The vLLM SERVER runs from the
# official vllm/vllm-openai image (see docker-compose.yml) — this image only
# needs the lightweight client-side deps, so it builds fast and runs anywhere.
FROM python:3.11-slim

WORKDIR /app

# System deps for matplotlib fonts etc. are bundled in the slim wheels; nothing else needed.
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir -e .

# Default: wait for the vLLM service then run the AWS sweep + report.
ENTRYPOINT ["gpubench"]
CMD ["run", "--config", "configs/aws.yaml"]
