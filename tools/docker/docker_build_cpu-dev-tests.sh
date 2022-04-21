echo docker build \
--pull \
--cache-from $CI_REGISTRY_IMAGE:latest \
--label "org.opencontainers.image.title=$CI_PROJECT_TITLE" \
--label "org.opencontainers.image.url=$CI_PROJECT_URL" \
--label "org.opencontainers.image.created=$CI_JOB_STARTED_AT" \
--label "org.opencontainers.image.revision=$CI_COMMIT_SHA" \
--label "org.opencontainers.image.version=$CI_COMMIT_REF_NAME" \
--tag $CI_REGISTRY_IMAGE:$CI_COMMIT_REF_NAME \
--build-arg OTBTESTS=true \
--build-arg BZL_CONFIGS="" \
--build-arg BASE_IMAGE="ubuntu:20.04" \
.