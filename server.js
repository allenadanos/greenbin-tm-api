const express = require('express');
const multer = require('multer');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 }
});

let model;
let metadata;

async function loadModel() {
  try {
    const modelPath = path.join(__dirname, 'model', 'model.json');
    const metadataPath = path.join(__dirname, 'model', 'metadata.json');

    console.log('Loading model from:', modelPath);
    model = await tf.loadLayersModel(`file://${modelPath}`);

    console.log('Loading metadata from:', metadataPath);
    const metadataContent = fs.readFileSync(metadataPath, 'utf8');
    metadata = JSON.parse(metadataContent);

    console.log('Model loaded successfully');
    console.log('Labels:', metadata.labels);

    console.log('Warming up model...');
    const warmupTensor = tf.zeros([1, 224, 224, 3]);
    model.predict(warmupTensor);
    warmupTensor.dispose();
    console.log('Model warmed up');

  } catch (error) {
    console.error('Error loading model:', error);
    throw error;
  }
}

async function preprocessImage(buffer) {
  const image = await tf.node.decodeImage(buffer, 3);
  const resized = tf.image.resizeBilinear(image, [224, 224]);
  const normalized = resized.div(255.0);
  const batched = normalized.expandDims(0);

  image.dispose();
  resized.dispose();
  normalized.dispose();

  return batched;
}

app.get('/', (req, res) => {
  res.json({
    service: 'GreenBin Waste Classification API',
    version: '1.0.0',
    endpoints: {
      health: 'GET /health',
      predict: 'POST /predict',
      info: 'GET /info'
    }
  });
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', modelLoaded: !!model });
});

app.get('/info', (req, res) => {
  if (!metadata) {
    return res.status(503).json({ error: 'Model not loaded' });
  }

  res.json({
    modelName: metadata.modelName,
    labels: metadata.labels,
    imageSize: metadata.imageSize,
    tfjsVersion: metadata.tfjsVersion,
    tmVersion: metadata.tmVersion
  });
});

app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    if (!model) {
      return res.status(503).json({ error: 'Model not loaded' });
    }

    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    console.log(`Processing image: ${req.file.originalname}, size: ${req.file.size} bytes`);

    const inputTensor = await preprocessImage(req.file.buffer);

    const predictions = model.predict(inputTensor);
    const probabilities = await predictions.data();

    inputTensor.dispose();
    predictions.dispose();

    const results = metadata.labels.map((label, index) => ({
      label: label,
      confidence: probabilities[index]
    })).sort((a, b) => b.confidence - a.confidence);

    const topPrediction = results[0];

    console.log(`Prediction: ${topPrediction.label} (${(topPrediction.confidence * 100).toFixed(2)}%)`);

    res.json({
      prediction: topPrediction.label,
      confidence: topPrediction.confidence,
      allPredictions: results.map(r => ({
        ...r,
        confidencePercent: `${(r.confidence * 100).toFixed(2)}%`
      }))
    });

  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ error: 'Prediction failed', details: error.message });
  }
});

app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ error: 'Internal server error', details: err.message });
});

loadModel().then(() => {
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Health check: http://localhost:${PORT}/health`);
    console.log(`Prediction endpoint: http://localhost:${PORT}/predict`);
  });
}).catch(error => {
  console.error('Failed to start server:', error);
  process.exit(1);
});