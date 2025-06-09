import os
import subprocess
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import logging
import tempfile

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        """Initialize the audio processor with Whisper model."""
        logger.info("Initializing Whisper model...")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        logger.info("Whisper model initialized successfully")

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video using ffmpeg."""
        try:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_audio_path = temp_file.name

            # Extract audio using ffmpeg command
            command = [
                'ffmpeg',
                '-i', video_path,
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16k',
                '-y',
                temp_audio_path
            ]
            
            # Run ffmpeg command
            subprocess.run(command, check=True, capture_output=True)
            
            return temp_audio_path
        except Exception as e:
            logger.error(f"Failed to extract audio: {str(e)}")
            raise

    def transcribe_audio(self, audio_path: str) -> dict:
        """Transcribe audio using Whisper model."""
        try:
            # Read audio file
            with open(audio_path, 'rb') as f:
                audio_data = f.read()

            # Process audio
            input_features = self.processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features

            if torch.cuda.is_available():
                input_features = input_features.to("cuda")

            # Generate transcription
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            return {
                'text': transcription,
                'segments': [],  # Whisper base model doesn't provide segments
                'language': 'en'  # Default to English
            }
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {str(e)}")
            raise
        finally:
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def process_video_audio(self, video_path: str) -> dict:
        """Process video audio: extract and transcribe."""
        try:
            # Extract audio
            audio_path = self.extract_audio(video_path)
            
            # Transcribe audio
            result = self.transcribe_audio(audio_path)
            
            return result
        except Exception as e:
            logger.error(f"Failed to process video audio: {str(e)}")
            raise 