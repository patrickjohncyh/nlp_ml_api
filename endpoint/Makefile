.PHONY: build bash prod up-prod

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

create_app:
	heroku apps:create ${HEROKU_APP_NAME}
	heroku git:remote -a ${HEROKU_APP_NAME}
	heroku stack:set container

delete_app:
	heroku apps:destroy ${HEROKU_APP_NAME} --confirm ${HEROKU_APP_NAME}
	git remote rm heroku

deploy:
	git checkout master
	@if [ "`git branch --list ${HEROKU_APP_NAME}-endpoint`" ]; then \
		git branch -D ${HEROKU_APP_NAME}-endpoint; \
	fi
	git checkout -b ${HEROKU_APP_NAME}-endpoint
	git add model_deploy.pickle
	git commit -m "model deployment to heroku"
	cd ..; git push heroku `git subtree split --prefix endpoint ${HEROKU_APP_NAME}-endpoint`:master --force


	git checkout master
	git branch -D ${HEROKU_APP_NAME}-endpoint