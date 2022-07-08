mkdir -p data
mkdir -p melanoma_results

wget -Odata/ST_mel1_rep2_counts.tsv  'https://www.dropbox.com/s/aha4mcdrq12myfi/ST_mel1_rep2_counts.tsv?dl=0'

filter_bleed \
  --count-mat data/ST_mel1_rep2_counts.tsv \
  --data-dir melanoma_results \
  --n-gene 1000 \
  --filter-type spots

prepare_kfold \
  --data-dir melanoma_results \
  --count-mat data/ST_mel1_rep2_counts.tsv

deconvolve \
  --data-dir melanoma_results \
  --n-gene 1000 \
  --n-components 4 \
  --lam2 10000 \
  --n-samples 100 \
  --n-burnin 500 \
  --n-thin 5

spatial_expression \
  --data-dir melanoma_results \
  --n-spatial-patterns 10 \
  --n-samples 100 \
  --n-burn 100 \
  --n-thin 2 \
  --simple