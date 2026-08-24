#!/usr/bin/env bash
# Code Ocean capsule setup: pull code+data from GitHub into /code and /data
set -e
cd /tmp && rm -rf cf
if command -v git >/dev/null 2>&1; then
    git clone --depth 1 https://github.com/yangbin-xd/cell-free.git cf
else
    curl -fL https://github.com/yangbin-xd/cell-free/archive/refs/heads/main.tar.gz | tar xz
    mv cell-free-main cf
fi
rm -rf cf/.git
find /data -mindepth 1 -delete 2>/dev/null || true
cp -a cf/data/. /data/
rm -rf cf/data
find /code -mindepth 1 -delete 2>/dev/null || true
cp -a cf/. /code/
chmod +x /code/run
echo "== /code ==" && ls /code
echo "== /data ==" && ls /data
echo "SETUP OK"
