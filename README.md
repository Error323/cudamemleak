# CUDA Memory leak
This program exposes a memory leak in a cuda program when compiled and run on a
Jetson TK1. For viewing the leak you should run the command below while the
program is active with the given $PID.

    sudo echo 0 $(awk '/Private/ {print "+", $2}' /proc/$PID/smaps) | bc
