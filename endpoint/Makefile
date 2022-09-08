.PHONY: build dev test tests up-dev bash prod up-dev up-prod

build: prod

prod:
	docker-compose build prod

up-prod:
	docker run --name endpoint -p 80:80 endpoint/endpoint:latest-prod

down-prod:
	docker kill endpoint
	docker rm endpoint

bash:
	docker-compose run --entrypoint=bash prod

deploy:
	git checkout -b endpoint
	git add model_deploy.pickle
	git commit -m "add model for deployment"
	git push heroku `git subtree split --prefix endpoint endpoint`:master --force
	git branch -D endpoint
