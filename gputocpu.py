import cv2
import easyocr
import matplotlib.pyplot as plt
import time

reader = easyocr.Reader(['en'], gpu=False)

video_path = 'img.png'
cap = cv2.VideoCapture(video_path)

frame_rate = 5
frame_count = 0

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  

    if frame_count % frame_rate == 0:
        results = reader.readtext(frame)

        for bbox, text, score in results:
            if score > 0.25:  
                cv2.rectangle(frame, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 2)
                cv2.putText(frame, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    frame_count += 1


cap.release()


end_time = time.time()
elapsed_time = end_time - start_time

if elapsed_time > 0:
    fps = frame_count / elapsed_time
else:
    fps = 0  

print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds.")
print(f"FPS: {fps:.2f}")

