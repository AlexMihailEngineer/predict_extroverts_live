# Personality Prediction API

This project is a FastAPI-based web application that predicts whether a person is an extrovert or an introvert based on a set of input features. The prediction is made by a machine learning model with 0.9733 precision trained on a quality dataset of 20_000 rows from the Kaggle competition [Predict the Introverts from the Extroverts](https://www.kaggle.com/competitions/playground-series-s5e7)

## Features

-   **Personality Prediction**: Predicts personality (Extrovert/Introvert) based on user input.
-   **FastAPI Backend**: A robust and fast backend serving the machine learning model.
-   **Static Frontend**: A simple HTML frontend to interact with the API.

## API Endpoint

The main API endpoint for prediction is `/predict`.

### Input

The `/predict` endpoint expects a JSON payload with the following structure:

```json
{
  "Time_spent_Alone": 0.0,
  "Stage_fear": "No",
  "Social_event_attendance": 6.0,
  "Going_outside": 4.0,
  "Drained_after_socializing": "No",
  "Friends_circle_size": 15.0,
  "Post_frequency": 5.0
}
```

### Output

The API returns a JSON object with the predicted personality:

```json
{
  "Personality": "Extrovert"
}
```

## How to Run

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the application**:
    ```bash
    uvicorn app.main:app --reload --port 80
    ```
3.  **Access the application**:
    Open your browser and navigate to `http://127.0.0.1/`.

## OR
access the application [online](http://51.20.62.183/) 

## Project Structure

```
.
├── app
│   ├── __init__.py
│   └── main.py
├── models
│   ├── ... (machine learning models)
├── static
│   └── index.html
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```