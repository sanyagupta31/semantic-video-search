import chromadb
from sentence_transformers import SentenceTransformer
from processor import process_video # Importing your code from Day 1

# 1. Initialize the AI Model (again, for text)
print("Loading model...")
model = SentenceTransformer('clip-ViT-B-32')

# 2. Initialize the Vector Database (ChromaDB)
# This creates a folder named 'my_video_db' to store data
client = chromadb.PersistentClient(path="my_video_db")

# Delete old collection if it exists (fresh start)
try:
    client.delete_collection(name="video_frames")
except:
    pass

collection = client.create_collection(name="video_frames")

# --- STEP 1: ADD DATA TO DATABASE ---
print("Extracting frames from video...")
video_data = process_video("test_video.mp4")

print(f"Storing {len(video_data)} frames in ChromaDB...")

# Prepare data for ChromaDB
ids = [str(i) for i in range(len(video_data))]  # ["0", "1", "2"...]
embeddings = [item['embedding'] for item in video_data]
metadatas = [{"timestamp": item['timestamp']} for item in video_data]

# Save to DB
collection.add(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas
)
print("✅ Data stored! Database is ready.")

# --- STEP 2: SEARCH FUNCTION ---
def search_video(query_text):
    print(f"\n🔍 Searching for: '{query_text}'")
    
    # 1. Convert text to numbers
    query_embedding = model.encode([query_text]).tolist()
    
    # 2. Search the database for the top 3 matches
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    
    # 3. Print results
    best_timestamp = results['metadatas'][0][0]['timestamp']
    score = results['distances'][0][0] # Lower distance = better match
    
    print(f"🎯 Best Match Found at: {best_timestamp} seconds")
    print(f"   (Confidence Score: {score})")
    return best_timestamp

# --- TEST IT ---
search_video("a blue circle")
search_video("a red square") # Should give a bad match or different timestamp