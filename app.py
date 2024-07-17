from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

def detect_finger_state(hand_landmarks):
    """
    Detect the state of the finger (open or closed) based on landmarks.
    Returns True if finger is open, False otherwise.
    """
    if not hand_landmarks:
        return None

    # Thumb tip coordinates
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    # Index tip coordinates
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    # Check if thumb and index finger are close to each other
    thumb_index_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    finger_open = thumb_index_distance > 0.1

    return finger_open

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                finger_open = detect_finger_state(hand_landmarks)
                if finger_open is not None:
                    cv2.putText(frame, f'Finger Open: {finger_open}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Variable to track finger state changes
last_finger_open = None

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/finger_position')
def finger_position():
    global last_finger_open
    success, frame = cap.read()
    if not success:
        return jsonify({"x": None, "y": None, "drawing": False})

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            finger_open = detect_finger_state(hand_landmarks)
            x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]
            y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]
            
            # Check for two consecutive finger open events
            if last_finger_open is not None and last_finger_open and finger_open:
                drawing = True
            elif last_finger_open is not None and last_finger_open and not finger_open:
                drawing = False
            else:
                drawing = False
            
            last_finger_open = finger_open

            return jsonify({"x": x, "y": y, "drawing": drawing})
    return jsonify({"x": None, "y": None, "drawing": False})

if __name__ == '__main__':
    app.run(debug=True)
