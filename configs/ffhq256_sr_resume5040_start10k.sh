JOB_NAME=stylenerf-ffhq256-sr

DATA="openmmlab:s3://openmmlab/datasets/editing/ffhq/ffhq_imgs/ffhq_256/"
SPEC=paper256
MODEL=stylenerf_ffhq_fullSR_scale2
DESC=resume-5040-FULLSR-scale2-start10k
PYTHON_ARGS=(
    'resume=modelzoo/NeRF-GAN_StyleNeRF_00000--paper256-stylenerf_ffhq-noaug-FULLSR-scale2_network-snapshot-005040.pkl'
    'model.loss_kwargs.sr_curriculum=[10000,20000]'
)