import cv2

# Load the image
image_path = "./static/person.jpg"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

# Extract faces
for i, (x, y, w, h) in enumerate(faces):
    face = image[y:y+h, x:x+w]
    cv2.imwrite(f"face_{i+1}.jpg", face)
    print(f"Saved face_{i+1}.jpg")

print("Extraction complete.")