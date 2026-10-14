# Emotion Prediction

A web app that predicts the emotion in a sentence using a **Bidirectional GRU (BiGRU)** deep learning model. It is served with **FastAPI** and includes a simple browser UI.

The model classifies text into 6 emotions:

| Emotion | Emoji |
|---|---|
| sadness | 😢 |
| joy | 😄 |
| love | ❤️ |
| anger | 😠 |
| fear | 😨 |
| surprise | 😲 |

## Features

- Browser UI to type a sentence and see the predicted emotion
- REST API (`/predict`) returning the top emotion, confidence and the probability of every emotion
- Health check endpoint (`/health`)
- Auto-generated API docs (`/docs`)
- Pre-trained model included, so no training is needed to run the app

## Tech Stack

- **Model:** TensorFlow / Keras (stacked Bidirectional GRU, 300-d embeddings)
- **Backend:** FastAPI + Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Dataset:** [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) (Hugging Face)

## Project Structure

```
Emotion-Prediction/
├── main.py                  # FastAPI app (API + serves the UI)
├── Artifacts/
│   ├── BiGRU_Model.keras    # Trained BiGRU model
│   └── tokenizer.pkl        # Fitted tokenizer
├── static/
│   ├── index.html           # Web UI
│   ├── style.css
│   └── script.js
├── final_clean.ipynb        # Training notebook (RNN, LSTM, GRU, BiGRU comparison)
├── requirements.txt
└── runtime.txt              # Python version (3.11.9)
```

## Getting Started

### Prerequisites

- **Python 3.11** (3.9 to 3.12 work with TensorFlow 2.17; Python 3.13 is **not** supported by the pinned versions)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/rithish1233/Emotion-Prediction.git
cd Emotion-Prediction

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

> On Windows with multiple Python versions, create the venv with `py -3.11 -m venv venv`.

### Run the app

Run this from the project root folder (the app loads files using relative paths):

```bash
uvicorn main:app --reload
```

Wait for `Model are loaded successfully...` in the console, then open:

- **Web UI:** http://127.0.0.1:8000
- **API docs:** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health

## API Usage

### `POST /predict`

Request:

```json
{
  "text": "I feel so happy and excited"
}
```

`text` must be between 1 and 2000 characters.

Example with curl:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so scared right now"}'
```

Response:

```json
{
  "text": "I am so scared right now",
  "predicted_emotion": "fear",
  "confidence": 0.93,
  "all_probabilites": {
    "sadness": 0.01,
    "joy": 0.01,
    "love": 0.00,
    "anger": 0.03,
    "fear": 0.93,
    "surprise": 0.02
  }
}
```

(The numbers above are illustrative.)

### `GET /health`

```json
{
  "status": "Server is running",
  "model_loaded": true
}
```

## How It Works

1. The input text is cleaned: lowercased, apostrophes removed, special characters stripped, extra spaces collapsed.
2. The tokenizer converts words to integer sequences.
3. Sequences are padded or truncated to **50 tokens**.
4. The BiGRU model outputs probabilities for the 6 emotions.
5. The API returns the top emotion and the full probability breakdown.

## Model

Training is documented in `final_clean.ipynb`:

- **Phase 1:** compares plain SimpleRNN, LSTM and GRU models
- **Phase 2:** builds the final stacked Bidirectional GRU (300-d embeddings, dropout 0.5)
- Uses balanced class weights to handle the dataset's class imbalance
- Includes confusion matrices and bias analysis

To retrain, open the notebook in Jupyter or Google Colab. It needs extra packages that are not in `requirements.txt` (for example `datasets`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`).

## Limitations

- Only 6 emotion classes, with no "neutral" class, so neutral text is still assigned one of the six
- Only the first 50 tokens of the input are used
- Trained on short, tweet-style text, so it works best on sentences like "I feel ..."
- Sarcasm, negation and mixed emotions can be misclassified
- `love` and `surprise` are smaller classes in the dataset and are harder to predict

## License

Add a license of your choice (for example MIT) if you want others to reuse this project.
