#!/bin/bash

set -e

MAIN_DIR="$(dirname $(realpath $0) | sed -e 's/\utils//')"

cd $MAIN_DIR
cd src

TYPE1=touch

echo "demo"
python demo.py
mkdir -p ${TYPE1}
rm -f ${TYPE1}/*
N=$(echo "$(ls demo_${TYPE1}/demo_*| wc -l) -1"| bc)
for i in $(seq 0 $N); do
  echo "convert frame $i"
  n=$(echo $i |xargs printf "%08d")
  for type in $TYPE1; do
    mogrify -rotate 180 demo_${type}/demo_${type}${n}.png
    convert -scale 400 -antialias  \
    demo_${type}/demo_${type}${n}.png \
    gen_proc_${type}/gen_proc_${type}${n}.png \
    gen_mod_${type}/gen_mod_${type}${n}.png \
    prederr_${type}/prederr_${type}${n}.png \
    -append ${type}/${type}${n}.png
  done
done

# screenshots
for type in $TYPE1; do
  cp ${type}/${type}*${N}.png ${MAIN_DIR}/pics/${type}.png
done

echo "videos"
for type in $TYPE1; do
  convert -loop 0 -delay 10 ${type}/${type}* ${MAIN_DIR}/pics/${type}.gif
  convert -loop 0 -delay 10 demo_${type}/demo_${type}* ${MAIN_DIR}/pics/demo_${type}.gif
  convert -loop 0 -delay 10 gen_mod_${type}/gen_mod_${type}* ${MAIN_DIR}/pics/gen_mod_${type}.gif
  convert -loop 0 -delay 10 gen_proc_${type}/gen_proc_${type}* ${MAIN_DIR}/pics/gen_proc_${type}.gif
  convert -loop 0 -delay 10 prederr_${type}/prederr_${type}* ${MAIN_DIR}/pics/prederr_${type}.gif
done

echo "clear"
rm -r gen_* demo_* $TYPE1 prederr_*
