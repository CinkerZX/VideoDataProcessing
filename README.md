# VideoDataProcessing

This is repo is for processing the video data of the power dynamic project



---

Steps:

- Generate subtitles | time | subtitles | (whisper)

- Assign labels: **gender**, **role**
  
  - split the audio based on timestamps (Pydub)
  
  - extract features, MFCCs (librosa)
  
  - clustering (scikit-learn / pyAudioAnalysis) - role
  
  - gender detection (pyAudioAnalysis) - gender
* Assign labels: **claim** / **grant**; **process** / **task**
  
  * Employee local deepseek model
