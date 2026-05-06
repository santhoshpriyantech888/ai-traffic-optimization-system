import cv2

# Load pre-trained car detection model
car_cascade = cv2.CascadeClassifier('cars.xml')

# Load video
cap = cv2.VideoCapture('sample_video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    # Draw rectangles around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # Count vehicles
    vehicle_count = len(cars)

    # Display traffic level
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Simulated signal logic
    if vehicle_count > 10:
        signal = "GREEN (High Traffic)"
    else:
        signal = "RED (Low Traffic)"

    cv2.putText(frame, f"Signal: {signal}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Traffic Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
