# Fitness Assistant
This is user friendly fitness assistant, which is RAG applications built as a part of LLM Zoomcamp

<p align="center">
  <img src="anime.png">
</p>

## Project overview

The Fitness Assistant is a RAG application designed to assist
users with their fitness routines.

The main use cases include:

1. Exercise Selection: Recommending exercises based on the type
of activity, targeted muscle groups, or available equipment.
2. Exercise Replacement: Replacing an exercise with suitable
alternatives.
3. Exercise Instructions: Providing guidance on how to perform a
specific exercise.
4. Conversational Interaction: Making it easy to get information
without sifting through manuals or websites.

## Dataset

The dataset used in this project contains information about
various exercises, including:

- **Exercise Name:** The name of the exercise (e.g., Push-Ups, Squats).
- **Type of Activity:** The general category of the exercise (e.g., Strength, Mobility, Cardio).
- **Type of Equipment:** The equipment needed for the exercise (e.g., Bodyweight, Dumbbells, Kettlebell).
- **Body Part:** The part of the body primarily targeted by the exercise (e.g., Upper Body, Core, Lower Body).
- **Type:** The movement type (e.g., Push, Pull, Hold, Stretch).
- **Muscle Groups Activated:** The specific muscles engaged during
the exercise (e.g., Pectorals, Triceps, Quadriceps).
- **Instructions:** Step-by-step guidance on how to perform the
exercise correctly.

The dataset was generated using ChatGPT and contains 207 records. It serves as the foundation for the Fitness Assistant's exercise recommendations and instructional support.

You can find the data in [`data/data.csv`](data/data.csv).





## Running the application


Running the Flask application Locally do this:

```bash
pipenv run python app.py
```

## Preparing the application

Before we can use the app, we need to initialize the database.
We can do it by running [`db_prep.py`](fitness_assistant/db_prep.py) script:

```bash
cd fitness_asstant
export POSTGRES_HOST=localhost
pipenv run python db_prep.py
```

Testing the Flask App with curl command:
```bash

URL=http://127.0.0.1:5000

DATA='{
    "question": "Is the Lat Pulldown considered a strength training activity, and if so why?"
}'

curl -X POST \
    -H "Content-Type: application/json" \
    -d "${DATA}" \
    ${URL}/question

```


Alternatively You can test the application by running test.py

```bash
pipenv run python test.py
```

Sending feedback:

```bash

ID="0fcc38c8-b2ef-4712-9497-ecceee72dd11"
URL=http://localhost:5000
FEEDBACK_DATA='{
    "conversation_id": "'${ID}'",
    "feedback": 1
}'

curl -X POST \
    -H "Content-Type: application/json" \
    -d "${FEEDBACK_DATA}" \
    ${URL}/feedback
```



## Runnning with the Docker

The easisest way to run the application is with the docker
```bash
docker-compose up
```


We use pipenv for managing dependencies and pyhton 3.11.
make sure you have pipenv installed:


```bash
pip install pipenv 
```

Running Jupyter notebook for experiments:

```bash
cd notebooks
pipenv run jupyter notebook
``` 




#### 1. create virtual enviournment
```bash
    pipenv install openai scikit-learn pandas flask
    pienv install --dev tqdm notebook==7.1.2 ipywidgets
    activate virtul enviournment: pipenv shell
```
