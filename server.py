from flask import Flask
from flask_socketio import SocketIO, emit
import numpy as np
import cv2
import base64
import face_recognition

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return "El servidor de detección de rostros está corriendo."

def detect_faces_in_frame(img):
    # Detecta los rostros en la imagen
    face_locations = face_recognition.face_locations(img)
    
    # Dibuja los recuadros alrededor de los rostros detectados
    for top, right, bottom, left in face_locations:
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    return img, face_locations

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

        img = cv2.flip(img, 1)

        processed_img, face_locations = detect_faces_in_frame(img)

        # Codifica la imagen procesada para enviarla de vuelta
        _, buffer = cv2.imencode('.jpg', processed_img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        print("Procesamiento completado. Enviando respuesta...")
        emit('response', {'image': encoded_image, 'face_locations': str(face_locations)})
    except Exception as e:
        print(f"Error procesando el cuadro: {str(e)}")
        emit('error', {'error': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')