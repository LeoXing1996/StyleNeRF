#!/usr/bin/env bash
set -x

PARTITION=mm_lol
DRAIN_NODE="SH-IDC1-10-142-4-150,SH-IDC1-10-142-4-159"

GPUS=${GPUS:-8}
SRUN_ARGS=${SRUN_ARGS:-""}

WORK_DIR=./out

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS} \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=40 \
    --kill-on-bad-exit=1 \
    -x ${DRAIN_NODE} \
    ${SRUN_ARGS} \
    python run_train.py --outdir=${WORK_DIR} --cfg=${CONFIG} --gpus=${GPUS} \
                        --slurm --batch ${BATCH} --kimg ${KIMG} --data ${DATA} --desc ${DESC} \
                        ${PYTHON_ARGS}