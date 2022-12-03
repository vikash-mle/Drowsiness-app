SET IMAGE_NAME="drowsiness-app-deployment"
SET GCP_PROJECT="mlops-c3-g12"
SET GCP_ZONE="us-central1-a"
SET GOOGLE_APPLICATION_CREDENTIALS=/secrets/deployment.json

cd ..
docker run --rm --name %IMAGE_NAME% -ti ^
           -v /var/run/docker.sock:/var/run/docker.sock ^
           --mount type=bind,source="%cd%\deployment",target=/app ^
           --mount type=bind,source="%cd%\secrets",target=/secrets ^
           --mount type=bind,source="C:\Users\USER\.ssh",target=/home/app/.ssh ^
           --mount type=bind,source="%cd%\api-service",target=/api-service ^
           --mount type=bind,source="%cd%\frontend",target=/frontend ^
           -e GOOGLE_APPLICATION_CREDENTIALS="%GOOGLE_APPLICATION_CREDENTIALS%" ^
           -e GCP_PROJECT="%GCP_PROJECT%" ^
           -e GCP_ZONE="%GCP_ZONE%" %IMAGE_NAME%
