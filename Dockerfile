FROM python:3.13-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . \
 && useradd --create-home --uid 10001 invisimark \
 && mkdir /work && chown invisimark /work

USER invisimark
# Jobs live in /work and are wiped at start-up and by a sweeper. Mount it as tmpfs or a scratch volume, not a shared folder.
ENV IM_WORK=/work
EXPOSE 8080
# One process only: jobs and the queue live in its memory, so do not run several workers or replicas behind one address.
CMD ["invisimark", "host", "--host", "0.0.0.0", "--port", "8080"]
