import streamlit as st
import cv2
import tempfile
import os
import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image

# Page Config
st.set_page_config(page_title="Video AI Search", page_icon="🔍")

st.title("🔍 Semantic Video Search Engine")
st.write("Upload a video and search inside it using natural language.")

# Load Model Once (Cache it so it doesn't reload every time)
@st.cache_resource
def load_model():
    return SentenceTransformer('clip-ViT-B-32')

model = load_model()

# Setup Database
client = chromadb.PersistentClient(path="my_video_db")

# Helper: Process Video
def process_and_index(video_path):
    # Reset DB
    try:
        client.delete_collection(name="video_frames")
    except:
        pass
    
    # Create fresh collection
    collection = client.create_collection(name="video_frames")

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Safety check for 0 FPS (prevents division by zero error)
    if fps == 0:
        fps = 30 

    frame_count = 0
    status_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ids = []
    embeddings = []
    metadatas = []
    
    st.write("Processing frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract 1 frame per second
        if frame_count % fps == 0: 
            timestamp = frame_count / fps
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            embedding = model.encode(pil_image)
            
            # Save frame to disk for display
            frame_filename = f"frame_{timestamp}.jpg"
            pil_image.save(frame_filename)
            
            ids.append(str(timestamp))
            embeddings.append(embedding.tolist())
            metadatas.append({"timestamp": timestamp, "filename": frame_filename})
        
        frame_count += 1
        if total_frames > 0:
            status_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    
    if len(ids) > 0:
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    
    return collection

# --- GUI LOGIC ---

uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    if st.button("🧠 Process Video (Index Frames)"):
        with st.spinner("Watching video..."):
            collection = process_and_index(video_path)
        st.success("Video Processed! AI is ready.")

    query = st.text_input("Search query (e.g., 'a blue circle', 'person running')")

    if query:
        try:
            # 1. Try to get the database
            collection = client.get_collection(name="video_frames")
            
            # 2. Search
            query_embedding = model.encode([query]).tolist()
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=1
            )
            
            # 3. Check if we found anything
            if not results['metadatas'][0]:
                st.warning("No matches found.")
            else:
                best_time = results['metadatas'][0][0]['timestamp']
                best_img = results['metadatas'][0][0]['filename']
                score = results['distances'][0][0]
                
                st.header(f"Found match at {best_time} seconds")
                st.image(best_img, caption=f"Confidence Distance: {score:.2f}")
                
        except Exception as e:
            # THIS FIXES YOUR CRASH
            st.error("⚠️ Database not found! Please click the 'Process Video' button first.")
            st.error(f"Error details: {e}")