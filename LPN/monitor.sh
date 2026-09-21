#!/bin/bash
PID=8067
while kill -0 $PID 2>/dev/null; do
    sleep 30
done
sudo shutdown -h now
