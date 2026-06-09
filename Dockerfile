FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/sample_prices.csv ./data/sample_prices.csv
COPY configs ./configs

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["var-engine"]
CMD ["--offline", "--output", "outputs/docker_demo"]
