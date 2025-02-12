#!/usr/bin/env python
# coding: utf-8

# # Download and set up Whisper
import ffmpeg
import subprocess

import whisper
import sys
import logging
from tqdm import tqdm
import os

# Set up logging to output to a file
log_file = "Whisper_transcription.log"
logging.basicConfig(
    filename = log_file,
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # 'w' to overwrite log file on each run, use 'a' to append
)

# Log a start message
logging.info("Starting Whisper transcription...")

# Load the Whisper model
logging.info("Loading whisper model...")
model = whisper.load_model("large-v2", device = "cuda") # GPU
logging.info("Whisper model complete.")

options = {
    "language": "en",  # Input language, if omitted is auto detected
    "task": "translate",  # Or "transcribe" if you just want transcription
    "beam_size": 5,  # Increase beam size for better accuracy
    "best_of": 5,  # Consider more hypotheses for better quality
    "temperature": 0.5,  # Adjust temperature for better results
    "suppress_tokens": "-1",  # Suppress unwanted tokens
}

# Set the root folder for .mp4 files
root_folder = "./video"

# Step 1: Calculate the number of .mp4 files
num_files = sum(
    1 for dirpath, dirnames, filenames in os.walk(root_folder) 
    for filename in filenames
    if filename.endswith(".mp4") and "ffmpeg/tests" not in dirpath
)

logging.info(f"Number of .mp4 files found: {num_files}")

# Step2: Transcribe the videos one by one
with tqdm(total=num_files, desc="Transcribing Files") as pbar:
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            # if filename.endswith(".mp4"):
            if filename.endswith(".mp4") and "ffmpeg/tests" not in dirpath:
                filepath = os.path.join(dirpath, filename).replace("\\", "/")
                if os.path.isfile(filepath):
                    logging.info(f"Transcribing: {filepath}") 
                    try:
                        result = model.transcribe(filepath, fp16=False, verbose=True, **options)

                        # Initialize an empty string to store the output in SRT format
                        srt_output = ""

                        # Iterate over the segments from the result to get timestamp info
                        for segment in result['segments']:
                            # Extract the start and end times
                            start_time = segment['start']
                            end_time = segment['end']
                            text = segment['text'].strip()
                        
                            # Convert seconds to hh:mm:ss,ms format for the timestamps
                            start_formatted = f"{int(start_time//60):02}:{int(start_time%60):02}.{int((start_time*1000)%1000):03}"
                            end_formatted = f"{int(end_time//60):02}:{int(end_time%60):02}.{int((end_time*1000)%1000):03}"
                        
                            # Append the transcription with the time info to the SRT output
                            srt_output += f"[{start_formatted} --> {end_formatted}]  {text}\n"

                        # Save the transcription with timestamps to a file
                        filename_no_ext = os.path.splitext(filename)[0]
                        with open(os.path.join(dirpath, filename_no_ext + '.txt'), 'w') as f:
                            f.write(srt_output)
                        logging.info(f"Transcription for {filepath} saved successfully.")
                    except Exception as e:
                        logging.info(f"Error transcribing {filepath}: {e}")
                        
                    pbar.update(1)
logging.info("Whisper transcription completed")




