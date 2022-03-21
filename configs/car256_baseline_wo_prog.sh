JOB_NAME=car256_noprog

DATA="s3://data/compCar256/"
SPEC=paper256
MODEL=stylenerf_cars
DESC=no_prog
PYTHON_ARGS="model.G_kwargs.synthesis_kwargs.progressive=False model.loss_kwargs.curriculum=None"