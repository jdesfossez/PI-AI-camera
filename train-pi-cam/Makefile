NR_EPOCH := 100
VAL_INTERVAL := 10

build:
	docker build --tag piaitrain .

train:
	docker run -v ${DATA_PATH}:/data --gpus all --shm-size=12g \
		-e NR_EPOCH=${NR_EPOCH} \
		-e VAL_INTERVAL=${VAL_INTERVAL} \
		piaitrain ./train-det.py
