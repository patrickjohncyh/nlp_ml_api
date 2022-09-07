.PHONY: build dev test tests up-dev bash prod up-dev up-prod

build: prod

prod:
	docker-compose build prod

up-prod:
	docker run --name endpoint -p 80:80 endpoint/endpoint:latest-prod

down-prod:
	docker rm endpoint

bash:
	docker-compose run --entrypoint=bash prod

