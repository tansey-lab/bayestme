

bsub -e error_KL_1.log -o output_KL_1.log -n 1 -W 4:00 -R 'span[hosts=1] rusage[mem=24]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_A1_IGO_06000_KL_1/outs:/input \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_A1_IGO_06000_KL_1/results:/data $(pwd)/bayestme_latest.sif \
  filter_genes --spaceranger /input --data-dir /data --n-top 1000 --filter-type ribosome

bsub -e error_KL_2.log -o output_KL_2.log -n 1 -W 4:00 -R 'span[hosts=1] rusage[mem=24]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_B1_IGO_06000_KL_2/outs:/input \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_B1_IGO_06000_KL_2/results:/data $(pwd)/bayestme_latest.sif \
  filter_genes --spaceranger /input --data-dir /data --n-top 1000 --filter-type ribosome

bsub -e error_KL_4.log -o output_KL_4.log -n 1 -W 4:00 -R 'span[hosts=1] rusage[mem=24]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-63205_6-7D_D1_IGO_06000_KL_4/outs:/input \
  --bind /work/tansey/bayestme/S19-63205_6-7D_D1_IGO_06000_KL_4/results:/data $(pwd)/bayestme_latest.sif \
  filter_genes --spaceranger /input --data-dir /data --n-top 1000 --filter-type ribosome

bsub -e error_KL_3.log -o output_KL_3.log -n 1 -W 4:00 -R 'span[hosts=1] rusage[mem=24]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-63205_6-7D_C1_IGO_06000_KL_3/outs:/input \
  --bind /work/tansey/bayestme/S19-63205_6-7D_C1_IGO_06000_KL_3/results:/data $(pwd)/bayestme_latest.sif \
  filter_genes --spaceranger /input --data-dir /data --n-top 1000 --filter-type ribosome



bsub -e error_KL_1_bleed.log -o output_KL_1_bleed.log -n 1 -W 24:00 -R 'span[hosts=1] rusage[mem=24]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_A1_IGO_06000_KL_1/outs:/input \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_A1_IGO_06000_KL_1/results:/data $(pwd)/bayestme_latest.sif \
  bleeding_correction --data-dir /data --n-top 50

bsub -e error_KL_2_bleed.log -o output_KL_2_bleed.log -n 1 -W 24:00 -R 'span[hosts=1] rusage[mem=24]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_B1_IGO_06000_KL_2/outs:/input \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_B1_IGO_06000_KL_2/results:/data $(pwd)/bayestme_latest.sif \
  bleeding_correction  --data-dir /data --n-top 50

bsub -e error_KL_4_bleed.log -o output_KL_4_bleed.log -n 1 -W 24:00 -R 'span[hosts=1] rusage[mem=24]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-63205_6-7D_D1_IGO_06000_KL_4/outs:/input \
  --bind /work/tansey/bayestme/S19-63205_6-7D_D1_IGO_06000_KL_4/results:/data $(pwd)/bayestme_latest.sif \
  bleeding_correction  --data-dir /data --n-top 50

bsub -e error_KL_3_bleed.log -o output_KL_3_bleed.log -n 1 -W 24:00 -R 'span[hosts=1] rusage[mem=24]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-63205_6-7D_C1_IGO_06000_KL_3/outs:/input \
  --bind /work/tansey/bayestme/S19-63205_6-7D_C1_IGO_06000_KL_3/results:/data $(pwd)/bayestme_latest.sif \
  bleeding_correction --data-dir /data --n-top 50


/opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_A1_IGO_06000_KL_1:/data \
  $(pwd)/bayestme_latest.sif \
  python -c "import bayestme.cli.plot_bleeding_correction; bayestme.cli.plot_bleeding_correction.main()" \
  --raw-stdata /data/raw_data_top_1000_std_95_spots.h5ad \
  --corrected-stdata /data/corrected_top_1000_std_95_spots.h5ad \
  --bleed-correction-results /data/top_1000_std_95_spots_bleed_correction.h5ad \
  --output-dir /data/results/cleaned_data_plots \
  --n-top 10

/opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_B1_IGO_06000_KL_2:/data \
  $(pwd)/bayestme_latest.sif \
  python -c "import bayestme.cli.plot_bleeding_correction; bayestme.cli.plot_bleeding_correction.main()" \
  --raw-stdata /data/raw_data_top_1000_std_95_spots.h5ad \
  --corrected-stdata /data/corrected_top_1000_std_95_spots.h5ad \
  --bleed-correction-results /data/top_1000_std_95_spots_bleed_correction.h5ad \
  --output-dir /data/results/cleaned_data_plots \
  --n-top 10

/opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-63205_6-7D_C1_IGO_06000_KL_3:/data \
  $(pwd)/bayestme_latest.sif \
  python -c "import bayestme.cli.plot_bleeding_correction; bayestme.cli.plot_bleeding_correction.main()" \
  --raw-stdata /data/raw_data_top_1000_std_95_spots.h5ad \
  --corrected-stdata /data/corrected_top_1000_std_95_spots.h5ad \
  --bleed-correction-results /data/top_1000_std_95_spots_bleed_correction.h5ad \
  --output-dir /data/results/cleaned_data_plots \
  --n-top 10

/opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-63205_6-7D_D1_IGO_06000_KL_4:/data \
  $(pwd)/bayestme_latest.sif \
  python -c "import bayestme.cli.plot_bleeding_correction; bayestme.cli.plot_bleeding_correction.main()" \
  --raw-stdata /data/raw_data_top_1000_std_95_spots.h5ad \
  --corrected-stdata /data/corrected_top_1000_std_95_spots.h5ad \
  --bleed-correction-results /data/top_1000_std_95_spots_bleed_correction.h5ad \
  --output-dir /data/results/cleaned_data_plots \
  --n-top 10


bsub -e error_KL_1_prepare_kfold.log -o error_KL_1_prepare_kfold.log -n 1 -W 4:00 -R 'span[hosts=1] rusage[mem=16]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_A1_IGO_06000_KL_1/results_no_bleed_correction:/data $(pwd)/bayestme_latest.sif \
  prepare_kfold --data-dir /data


bsub -e error_KL_2_prepare_kfold.log -o error_KL_2_prepare_kfold.log -n 1 -W 4:00 -R 'span[hosts=1] rusage[mem=16]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_B1_IGO_06000_KL_2/results_no_bleed_correction:/data $(pwd)/bayestme_latest.sif \
  prepare_kfold --data-dir /data

bsub -e error_KL_3_prepare_kfold.log -o error_KL_3_prepare_kfold.log -n 1 -W 4:00 -R 'span[hosts=1] rusage[mem=16]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-63205_6-7D_C1_IGO_06000_KL_3/results_no_bleed_correction:/data $(pwd)/bayestme_latest.sif \
  prepare_kfold --data-dir /data

bsub -e error_KL_4_prepare_kfold.log -o error_KL_4_prepare_kfold.log -n 1 -W 4:00 -R 'span[hosts=1] rusage[mem=16]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-63205_6-7D_D1_IGO_06000_KL_4/results_no_bleed_correction:/data $(pwd)/bayestme_latest.sif \
  prepare_kfold --data-dir /data


bsub -e error_KL_3_prepare_kfold.log -o error_KL_3_prepare_kfold.log -n 1 -W 4:00 -R 'span[hosts=1] rusage[mem=16]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-63205_6-7D_C1_IGO_06000_KL_3/results:/data $(pwd)/bayestme_latest.sif \
  prepare_kfold --data-dir /data


bsub -e KL_1_deconvolve.err -o KL_1_deconvolve.log -n 1 \
  -W 96:00 \
  -R 'span[hosts=1] rusage[mem=64]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_A1_IGO_06000_KL_1:/data \
  $(pwd)/bayestme_latest.sif \
  deconvolve \
  --input /data/corrected_top_1000_std_95_spots.h5ad \
  --output /data/deconvolve_results.h5ad \
  --n-gene 1000 \
  --lam2 1000 \
  --n-samples 100 \
  --n-burnin 500 \
  --n-thin 2 \
  --n-components 4


bsub -e KL_2_deconvolve.err -o KL_2_deconvolve.log -n 1 \
  -W 96:00 \
  -R 'span[hosts=1] rusage[mem=64]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_B1_IGO_06000_KL_2:/data \
  $(pwd)/bayestme_latest.sif \
  deconvolve \
  --input /data/corrected_top_1000_std_95_spots.h5ad \
  --output /data/deconvolve_results.h5ad \
  --n-gene 1000 \
  --lam2 1000 \
  --n-samples 100 \
  --n-burnin 500 \
  --n-thin 2 \
  --n-components 3

bsub -e KL_3_deconvolve.err -o KL_3_deconvolve.log -n 1 \
  -W 96:00 \
  -R 'span[hosts=1] rusage[mem=64]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-63205_6-7D_C1_IGO_06000_KL_3:/data \
  $(pwd)/bayestme_latest.sif \
  deconvolve \
  --input /data/corrected_top_1000_std_95_spots.h5ad \
  --output /data/deconvolve_results.h5ad \
  --n-gene 1000 \
  --lam2 1000 \
  --n-samples 100 \
  --n-burnin 500 \
  --n-thin 2 \
  --n-components 5

bsub -e KL_4_deconvolve.err -o KL_4_deconvolve.log -n 1 \
  -W 96:00 \
  -R 'span[hosts=1] rusage[mem=64]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-63205_6-7D_D1_IGO_06000_KL_4:/data \
  $(pwd)/bayestme_latest.sif \
  deconvolve \
  --input /data/corrected_top_1000_std_95_spots.h5ad \
  --output /data/deconvolve_results.h5ad \
  --n-gene 1000 \
  --lam2 1000 \
  --n-samples 100 \
  --n-burnin 500 \
  --n-thin 2 \
  --n-components 5


bsub -e KL_1_spatial_expression.err -o KL_1_spatial_expression.log -n 1 \
  -W 96:00 \
  -R 'span[hosts=1] rusage[mem=64]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_A1_IGO_06000_KL_1:/data \
  $(pwd)/bayestme_latest.sif \
  spatial_expression \
  --deconvolve-results /data/deconvolve_results.h5ad \
  --dataset /data/corrected_top_1000_std_95_spots.h5ad \
  --output /data/sde_results.h5ad \
  --n-gene 1000 \
  --n-samples 100 \
  --n-burn 100 \
  --n-thin 2 \
  --n-spatial-patterns 10 \
  --simple

bsub -e KL_2_spatial_expression.err -o KL_2_spatial_expression.log -n 1 \
  -W 96:00 \
  -R 'span[hosts=1] rusage[mem=64]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_B1_IGO_06000_KL_2:/data \
  $(pwd)/bayestme_latest.sif \
  spatial_expression \
  --deconvolve-results /data/deconvolve_results.h5ad \
  --dataset /data/corrected_top_1000_std_95_spots.h5ad \
  --output /data/sde_results.h5ad \
  --n-gene 1000 \
  --n-samples 100 \
  --n-burn 100 \
  --n-thin 2 \
  --n-spatial-patterns 10 \
  --simple

bsub -e KL_3_spatial_expression.err -o KL_3_spatial_expression.log -n 1 \
  -W 96:00 \
  -R 'span[hosts=1] rusage[mem=64]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-63205_6-7D_C1_IGO_06000_KL_3:/data \
  $(pwd)/bayestme_latest.sif \
  spatial_expression \
  --deconvolve-results /data/deconvolve_results.h5ad \
  --dataset /data/corrected_top_1000_std_95_spots.h5ad \
  --output /data/sde_results.h5ad \
  --n-gene 1000 \
  --n-samples 100 \
  --n-burn 100 \
  --n-thin 2 \
  --n-spatial-patterns 10 \
  --simple

bsub -e KL_4_spatial_expression.err -o KL_4_spatial_expression.log -n 1 \
  -W 96:00 \
  -R 'span[hosts=1] rusage[mem=64]' \
  /opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-63205_6-7D_D1_IGO_06000_KL_4:/data \
  $(pwd)/bayestme_latest.sif \
  spatial_expression \
  --deconvolve-results /data/deconvolve_results.h5ad \
  --dataset /data/corrected_top_1000_std_95_spots.h5ad \
  --output /data/sde_results.h5ad \
  --n-gene 1000 \
  --n-samples 100 \
  --n-burn 100 \
  --n-thin 2 \
  --n-spatial-patterns 10 \
  --simple

mkdir /work/tansey/bayestme/S19-76231_8-5_2_A1_IGO_06000_KL_1/results/sde_data_plots

/opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_A1_IGO_06000_KL_1:/data \
  $(pwd)/bayestme_latest.sif \
  python -c "import bayestme.cli.plot_spatial_expression; bayestme.cli.plot_spatial_expression.main()" \
  --stdata /data/corrected_top_1000_std_95_spots.h5ad \
  --deconvolution-result /data/deconvolve_results.h5ad \
  --sde-result /data/sde_results.h5ad \
  --output-dir /data/results/sde_data_plots

mkdir /work/tansey/bayestme/S19-76231_8-5_2_B1_IGO_06000_KL_2/results/sde_data_plots

/opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-76231_8-5_2_B1_IGO_06000_KL_2:/data \
  $(pwd)/bayestme_latest.sif \
  python -c "import bayestme.cli.plot_spatial_expression; bayestme.cli.plot_spatial_expression.main()" \
  --stdata /data/corrected_top_1000_std_95_spots.h5ad \
  --deconvolution-result /data/deconvolve_results.h5ad \
  --sde-result /data/sde_results.h5ad \
  --output-dir /data/results/sde_data_plots

mkdir /work/tansey/bayestme/S19-63205_6-7D_C1_IGO_06000_KL_3/results/sde_data_plots

/opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-63205_6-7D_C1_IGO_06000_KL_3:/data \
  $(pwd)/bayestme_latest.sif \
  python -c "import bayestme.cli.plot_bleeding_correction; bayestme.cli.plot_bleeding_correction.main()" \
  --stdata /data/corrected_top_1000_std_95_spots.h5ad \
  --deconvolution-result /data/deconvolve_results.h5ad \
  --sde-result /data/sde_results.h5ad \
  --output-dir /data/results/sde_data_plots

mkdir /work/tansey/bayestme/S19-63205_6-7D_D1_IGO_06000_KL_4/results/sde_data_plots

/opt/local/singularity/3.7.1/bin/singularity exec \
  --bind /work/tansey/bayestme/S19-63205_6-7D_D1_IGO_06000_KL_4:/data \
  $(pwd)/bayestme_latest.sif \
  python -c "import bayestme.cli.plot_bleeding_correction; bayestme.cli.plot_bleeding_correction.main()" \
  --stdata /data/corrected_top_1000_std_95_spots.h5ad \
  --deconvolution-result /data/deconvolve_results.h5ad \
  --sde-result /data/sde_results.h5ad \
  --output-dir /data/results/sde_data_plots
