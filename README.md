# GreenBin Waste Classification API

TensorFlow.js image classification API for waste sorting (Paper, Plastic, Biodegradable).

## Quick Start

```bash
npm install
npm start
```

API runs on `http://localhost:3000`

## API Endpoints

### `POST /predict`
Upload image for classification

**Request:**
```bash
curl -X POST http://localhost:3000/predict \
  -F "image=@path/to/image.jpg"
```

**Response:**
```json
{
  "prediction": "Paper",
  "confidence": 0.95,
  "allPredictions": [
    {"label": "Paper", "confidence": 0.95, "confidencePercent": "95.00%"},
    {"label": "Plastic", "confidence": 0.03, "confidencePercent": "3.00%"},
    {"label": "Biodegradable", "confidence": 0.02, "confidencePercent": "2.00%"}
  ]
}
```

### `GET /health`
Health check endpoint

### `GET /info`
Model information and labels

## Deployment

### Railway
1. Push code to GitHub
2. Create new project on railway.app
3. Select repository
4. Deploy automatically

### Render
1. Push code to GitHub
2. Create new Web Service on render.com
3. Connect repository
4. Deploy automatically

## Testing

```bash
# Health check
curl http://localhost:3000/health

# Classify image
curl -X POST http://localhost:3000/predict \
  -F "image=@test-image.jpg"
```