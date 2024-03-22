from flask import Flask
from flask_socketio import SocketIO, emit
import numpy as np
import cv2
import base64

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return "El servidor de detección de rostros está corriendo."

def detect_faces_in_frame(img):
    # Aplicar efecto espejo
    img = cv2.flip(img, 1)

    # Cargar varios clasificadores Haar
    cascades = ['haarcascade_frontalface_default.xml', 
                'haarcascade_frontalface_alt.xml', 
                'haarcascade_frontalface_alt2.xml', 
                'haarcascade_profileface.xml']
    
    faces_detected = []

    for cascade in cascades:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for face in faces:
            if face not in faces_detected:
                faces_detected.append(face)

    # Dibuja los recuadros alrededor de los rostros detectados en la imagen espejada
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img, faces_detected

@socketio.on('connect')
def handle_connect():
    print("Un cliente se ha conectado.")

@socketio.on('disconnect')
def handle_disconnect():
    print("Un cliente se ha desconectado.")

@socketio.on('frame')
def handle_frame(data):
    print("Recibiendo cuadro de video para procesamiento...")
    try:
        frame = data['image']
        if not frame.startswith('data:image/jpeg;base64,'):
            raise ValueError("Formato de imagen no esperado")

        # Elimina el prefijo y decodifica la imagen
        frame = frame.split(",")[1]
        frame = base64.b64decode(frame)
        nparr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        processed_img, faces = detect_faces_in_frame(img)

        # Codifica la imagen procesada para enviarla de vuelta
        _, buffer = cv2.imencode('.jpg', processed_img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        print("Procesamiento completado. Enviando respuesta...")
        emit('response', {'image': encoded_image, 'faces': str(faces)})
    except Exception as e:
        print(f"Error procesando el cuadro: {str(e)}")
        emit('error', {'error': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')
