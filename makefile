MAKES=serve test docker-build docker-up docker-down deploy undeploy


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
	gcloud run deploy chagas-ecg-net \
		--image=europe-west2-docker.pkg.dev/chagas-ecg-net/chagas-ecg-net/chagas-ecg-net:latest \
		--region=europe-west2 \
		--allow-unauthenticated \
		--port=8000 \
		--memory=2Gi \
		--set-env-vars="REGISTRY_ROOT=/app/model_registry,\
	MODEL_NAME=grouped_lead_cnn,\
	MODEL_VERSION=latest,\
	DEVICE=cpu,\
	CORS_ORIGINS=[\"*\"],\
	MAX_UPLOAD_BYTES=52428800,\
	FEEDBACK_DIR=/app/feedback"

undeploy:
	gcloud run services delete chagas-ecg-net --region=europe-west2