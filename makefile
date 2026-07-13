MAKES=serve test docker-up docker-build deploy


.PHONY: $(patsubst %,%,$(MAKES))

serve:
	uvicorn src.api.main:app --reload
	
test:
	pytest

docker-up:
	docker compose up --build

docker-build:
	docker build -t chagas-ecg-net:latest .

deploy:
# 	gcloud run deploy --region europe-west --allow-unauthenticated --port 8000 --memory 2Gi