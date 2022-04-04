JOB_NAME=stylenerf-ffhq256-sr

DATA="openmmlab:s3://openmmlab/datasets/editing/ffhq/ffhq_imgs/ffhq_256/"
SPEC=paper256
MODEL=stylenerf_ffhq_fullSR_scale2_start10k_w1e-1
DESC=resume-10080-FULLSR-scale2-start10k
PYTHON_ARGS='resume=modelzoo/StyleNeRF-FFHQ-256-Baseline-010080.pkl'