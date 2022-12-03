REM Define some environment variables
SET IMAGE_NAME=drowsiness-app-api-server

SET GCP_PROJECT="mlops-c3-group12"
SET GCP_ZONE="us-central1-a"
SET GOOGLE_APPLICATION_CREDENTIALS=/secrets/bucket-reader.json





REM Run the container
cd ..
docker run  --rm --name %IMAGE_NAME% -ti ^
            --mount type=bind,source="%cd%\api-service",target=/app ^
            --mount type=bind,source="%cd%\persistent-folder",target=/persistent ^
            --mount type=bind,source="%cd%\secrets",target=/secrets ^
            -e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS ^
            -e GCP_PROJECT=$GCP_PROJECT ^
            -e GCP_ZONE=$GCP_ZONE ^
            -p 9000:9000 -e DEV=0 %IMAGE_NAME%