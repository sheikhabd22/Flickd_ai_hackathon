# Vibe Classification Engine

A powerful AI-powered video processing engine that analyzes fashion videos to detect clothing items, match them with products, and classify the overall aesthetic vibe.

## Features

- **Video Processing**: Extracts frames and audio from uploaded videos
- **Fashion Item Detection**: Uses YOLOv8 to detect clothing and fashion accessories
- **Product Matching**: Matches detected items with products using CLIP embeddings
- **Vibe Classification**: Classifies the aesthetic style/vibe of the content
- **Audio Transcription**: Transcribes audio using Whisper for additional context
- **REST API**: FastAPI-based API for easy integration

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg installed on your system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flickd_ai_engine.git
cd flickd_ai_engine
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
- YOLOv8 model will be downloaded automatically on first run
- Other models will be downloaded automatically when needed

## Project Structure

```
flickd_ai_engine/
├── data/               # Data files including products.csv
├── frames/            # Temporary frame storage
├── images/            # Product images
├── models/            # Downloaded ML models and stores the embeddings 
├── outputs/           # Processing outputs
├── utils/             # Utility modules
├── main.py            # Main application code
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Usage

### Starting the API Server

```bash
python main.py
```

The server will start at `http://localhost:8000`

### API Endpoints

- `POST /process-video`: Process a video file
  - Input: Video file and optional caption
  - Output: JSON with detected items, matched products, and vibe classification

- `GET /health`: Health check endpoint

### Example API Usage

```python
import requests

url = "http://localhost:8000/process-video"
files = {"video": open("video.mp4", "rb")}
data = {"caption": "Optional video caption"}

response = requests.post(url, files=files, data=data)
results = response.json()
```

## Features in Detail

### Fashion Item Detection
- Uses YOLOv8 for object detection
- Detects 30+ fashion-related classes
- Configurable confidence thresholds

### Product Matching
- Uses CLIP embeddings for semantic matching
- FAISS index for efficient similarity search
- Matches detected items with product catalog

### Vibe Classification
- Classifies content into aesthetic categories:
  - Coquette
  - Clean Girl
  - Cottagecore
  - Streetcore
  - And more...

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- CLIP by OpenAI
- Whisper by OpenAI
- FAISS by Facebook Research 
