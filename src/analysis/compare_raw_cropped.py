import cv2
from src.preprocessing.face_detection import detect_and_crop_face

# Load image
image_path = "D:\\Deepfake Detection Model\\dataset\\real_vs_fake\\real-vs-fake\\test\\real\\00016.jpg"   # ← change this
image = cv2.imread(image_path)

# Save raw image
cv2.imwrite("raw_image.jpg", image)

# Detect and crop face
face_img, _ = detect_and_crop_face(image)

if face_img is not None:
    cv2.imwrite("cropped_face.jpg", face_img)
    print("Saved both images ✅")
else:
    print("No face detected ❌")