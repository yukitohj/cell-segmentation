IMAGE := "y-nagano-cell-segmentation"
CONTAINER := cell-segmentation

docker-build:
	docker build -t $(IMAGE) --build-arg USER_ID=$$(id -u) --build-arg GROUP_ID=$$(id -g) --build-arg WORKING_DIR=$$(pwd) .

docker-build-nocache:
	docker build -t $(IMAGE) --build-arg USER_ID=$$(id -u) --build-arg GROUP_ID=$$(id -g) --build-arg WORKING_DIR=$$(pwd)  --no-cache .

docker-run:
	docker run -it --rm -v $$(pwd):$$(pwd):rw -w $$(pwd) --gpus all --shm-size=1gb -p 5000:5000 --name $(CONTAINER) $(IMAGE)

docker-attach:
	docker attach $(CONTAINER)

docker-attach-as-root:
	docker exec -it -u root $(CONTAINER) /bin/bash
