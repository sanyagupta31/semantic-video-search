import cv2
import numpy as np

# Video settings
width, height = 640, 480
fps = 30
duration = 10  # seconds
output_file = 'test_video.mp4'

# Create the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

print(f"🎬 Generating {output_file}...")

# Generate frames
for i in range(fps * duration):
    # Create a black background
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw a moving blue circle
    center_x = int(i * (width / (fps * duration)))
    center_y = height // 2
    cv2.circle(frame, (center_x, center_y), 50, (255, 0, 0), -1)
    
    # Add text
    cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    out.write(frame)

out.release()
print(f"✅ Success! '{output_file}' has been created in your folder.")