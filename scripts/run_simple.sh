#!/bin/bash
#for i in {1..10}
for i in {1..1}
do
    echo "=============================="
    echo "사이클 $i 시작"
    ../CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low &
    sleep 5

    echo "사이클 $i: 데이터 캡처 시작"
    python samples/demo_show_image.py
    echo "사이클 $i: 데이터 캡처 완료"

    # kill Carla
	(sleep 3); kill -9 `ps | grep 'CarlaUE4-Linux-' | awk '{print $1}'`; (sleep 1)
    echo "사이클 $i 끝!"
    echo "=============================="
done
