ARCH=$1
ATTN=$2
STD=$3
GPU=$4
FILENAME=$5

for BENCHMARK in voc21 context60 coco_object voc20 city_scapes context59 ade20k coco_stuff164k
do
  printf "\n${BENCHMARK}, Arch: ${ARCH}, Attn: ${ATTN}, std: ${STD} \n\n" >> ${FILENAME}
  CUDA_VISIBLE_DEVICES=${GPU} python eval.py --config ./configs/cfg_${BENCHMARK}.py \
            --arch ${ARCH} \
            --attn ${ATTN} \
            --std ${STD} \
              |& tee -a ${FILENAME}
  printf "\n\n----------\n\n" >> ${FILENAME}
done