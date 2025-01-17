set -e # fail fast

docker buildx build . --platform linux/amd64 -t lossviz:latest


