FROM ubuntu:24.04
RUN apt-get update -qq && apt-get install -y -qq python3 python3-yaml perl libnuma1 libstdc++-13-dev libc6-dev >/dev/null && rm -rf /var/lib/apt/lists/*
