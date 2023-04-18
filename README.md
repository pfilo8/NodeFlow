
## How to build docker?
Copy project to the server.
```shell
make build
```

## How to run the container in the background?
```shell
make run
```

## How to get to the Docker container?
```shell
docker exec -it "${username}-${project_name}" /bin/bash
```

## How to run the Kedro project?
```shell
python -m kedro run --pipeline <pipeline_name>
```

## How to stop the container?
```shell
make rm
```
