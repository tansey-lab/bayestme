#!/bin/bash
code=$1
data_dir=$2

bsub -e ${code}_read_data.err -o ${code}_read_data.log -n 1 \
  -W 96:00 \
  -R 'span[hosts=1] rusage[mem=24]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/${data_dir}:/data \
  $(pwd)/bayestme_latest.sif \
  load_spaceranger \
  --input /data/outs \
  --output /data/raw_data.h5ad

bsub -e ${code}_filter_genes.err -o ${code}_filter_genes.log -n 1 \
  -W 96:00 \
  -R 'span[hosts=1] rusage[mem=24]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/${data_dir}:/data \
  $(pwd)/bayestme_latest.sif \
  filter_genes \
  --input /data/raw_data.h5ad \
  --output /data/raw_data_top_1000_std_95_spots.h5ad \
  --n-top-by-standard-deviation 1000 \
  --spot-threshold 0.95

bsub -e ${code}_bleeding_correction.err -o ${code}_bleeding_correction.log -n 1 \
  -W 96:00 \
  -R 'span[hosts=1] rusage[mem=24]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/${data_dir}:/data \
  $(pwd)/bayestme_latest.sif \
  bleeding_correction \
  --input /data/raw_data_top_1000_std_95_spots.h5ad \
  --bleed-out /data/top_1000_std_95_spots_bleed_correction.h5ad \
  --stdata-out /data/corrected_top_1000_std_95_spots.h5ad \
  --n-top 50

bsub -e ${code}_bleeding_correction.err -o ${code}_bleeding_correction.log -n 1 \
  -W 96:00 \
  -R 'span[hosts=1] rusage[mem=24]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/${data_dir}:/data \
  $(pwd)/bayestme_latest.sif \
  bleeding_correction \
  --input /data/corrected_top_1000_std_95_spots.h5ad \
  --bleed-out /data/top_1000_std_95_spots_bleed_correction.h5ad \
  --stdata-out /data/corrected_top_1000_std_95_spots.h5ad \
  --n-top 50



