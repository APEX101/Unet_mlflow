# the agenda on this dockerfile is pure trash and will not make ant sense
# create a linux image train the model on it 
# copy the artifacts to another instance of the same image to avoid biuld reqiurement size
# use that image to use the model and predict on a sample image
# create another stage with flask server on 
# create multi platform build
# create multi target build 
# use args and envs
# the second iamge will be a entrypoint executable

# creeate a python environemnt and create model artifacts
ARG PYTHON_VER=3.11.9
FROM python:$PYTHON_VER-alpine3.19 as trainbase
WORKDIR /traingrounds
RUN --mount=bind,target=./,source=./ \
    pip install -r reqiurements.txt
RUN python trainer.py

FROM python:$PYTHON_VER-alpine3.19
WORKDIR /predictgrounds
COPY */ ./ 
COPY --from=trainbase unet_membrane.keras ./
ENTRYPOINT [ "python","predictor.py"]


