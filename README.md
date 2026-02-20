# Clerk + Flask Application

This project consists of a Flask backend and a vanilla HTML/JS frontend.

## Prerequisites

- Python 3.9+
- Clerk Account (Publishable Key and Secret Key)

## Setup

1.  **Backend Dependencies**:
    ```bash
    pip install flask python-jose requests flask-cors
    ```

2.  **Configuration**:
    - **Backend**: Open `backend/auth.py` and update `CLERK_ISSUER` with your Clerk Issuer URL.
    - **Frontend**: Open `frontend/index.html` and update `data-clerk-publishable-key` with your Clerk Publishable Key.

## How to Run

You need to run the backend and frontend separately.

### 1. Start the Backend

Open a terminal at the project root (`e:\my projects\new football`) and run:

```bash
python backend/app.py
```
The backend will start on `http://localhost:5000`.

### 2. Start the Frontend

Do **not** use `node app.js` or `npm run dev`. This is a static site.

Open a **new terminal** window, go to the frontend directory, and use Python to serve it:

```bash
cd frontend
python -m http.server 8000
```

Then open your browser and go to: [http://localhost:8000](http://localhost:8000)

## Troubleshooting

- **`window is not defined`**: You are trying to run `app.js` with Node.js. It is a browser script and must be run in a browser (by loading `index.html`).
- **`npm error`**: This project does not use Node.js package manager (npm). It uses direct script tags.
