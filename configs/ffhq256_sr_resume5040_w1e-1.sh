JOB_NAME=stylenerf-ffhq256-sr

DATA="openmmlab:s3://openmmlab/datasets/editing/ffhq/ffhq_imgs/ffhq_256/"
SPEC=paper256
MODEL=stylenerf_ffhq_fullSR_scale2
DESC=-resume-5040-FULLSR-scale2-w1e-1
PYTHON_ARGS=(
    'resume=modelzoo/StyleNeRF-FFHQ-256-Baseline-005040.pkl'
    'model.loss_kwargs.sr_reg_weight=1e-2'
)