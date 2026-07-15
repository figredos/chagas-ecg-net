MAKES=serve test docker-build docker-up docker-down deploy


.PHONY: $(patsubst %,%,$(MAKES))

serve:
	uvicorn src.api.main:app --reload
	
test:
	pytest

docker-build:
	docker build -t chagas-ecg-net:latest .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

deploy:
# 	gcloud run deploy --region europe-west --allow-unauthenticated --port 8000 --memory 2Gi