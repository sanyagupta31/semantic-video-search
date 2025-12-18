import cv2
import os
from sentence_transformers import SentenceTransformer
from PIL import Image

# 1. Define the video path explicitly
video_filename = "test_video.mp4"

# Check if file exists before doing anything else
if not os.path.exists(video_filename):
    print(f"❌ ERROR: I cannot find the file '{video_filename}'")
    print(f"I am looking in this folder: {os.getcwd()}")
    print("Please make sure the video is in this exact folder.")
    exit()

print("Loading CLIP model...")
model = SentenceTransformer('clip-ViT-B-32')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Check if OpenCV can actually open it
    if not cap.isOpened():
        print(f"❌ ERROR: File exists, but OpenCV cannot open it. Codec issue?")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("❌ ERROR: FPS is 0. The video file might be corrupted.")
        return []

    print(f"✅ Video opened successfully! FPS: {fps}")
    
    frame_count = 0
    extracted_data = [] 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % int(fps) == 0:
            timestamp = frame_count / fps
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            embedding = model.encode(pil_image)
            
            extracted_data.append({
                "timestamp": timestamp,
                "embedding": embedding.tolist()
            })
            print(f"Processed timestamp: {timestamp:.2f}s")
            
        frame_count += 1

    cap.release()
    print(f"Finished! Extracted {len(extracted_data)} frames.")
    return extracted_data

# Run it
video_data = process_video(video_filename)
if len(video_data) > 0:
    print("SUCCESS! First 5 numbers:", video_data[0]['embedding'][:5])
else:
    print("No frames were extracted.")