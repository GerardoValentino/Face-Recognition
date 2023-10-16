import cv2
import os
import face_recognition
import pandas as pd
import csv
import datetime

'''
w_d = 'C:/Users/2109g/Desktop/Materias/TOPICOS/FaceRecognition/Data'
f_i = w_d + '/semana01.csv'

df = pd.read_csv(f_i)
'''

def set_attendance(day, attend_list):
    # Lee el archivo CSV existente y carga los datos en una lista de diccionarios
    with open('Data/semana01.csv', 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # Actualiza los valores en la lista de diccionarios según el día y la lista de asistencia
    for nombre in attend_list:
        for row in rows:
            if nombre in row['Nombre']:
                row[day] = 1  # Suponiendo que 1 representa asistencia
    
    # Escribe los cambios de vuelta al archivo CSV
    with open('Data/semana01.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fields = ['Nombre', 'Domingo', 'Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# Obtener la fecha y hora actual
fecha_actual = datetime.datetime.now()

# Obtener el día del mes
dia = fecha_actual.day

# Obtener el nombre del día de la semana
nombre_dia = fecha_actual.strftime("%A")

print("El día actual es:", nombre_dia, dia)

people = [{'Nombre':'Gerardo Valentino Rosales Ramos', 'Domingo': 0, 'Lunes': 0, 'Martes': 0, 'Miercoles': 0, 'Jueves': 0, 'Viernes': 0},
          {'Nombre': 'Eduardo Caudillo Gonzalez', 'Domingo': 0, 'Lunes': 0, 'Martes': 0, 'Miercoles': 0, 'Jueves': 0, 'Viernes': 0},
          {'Nombre': 'Akari Guadalupe Almanza Acosta', 'Domingo': 0, 'Lunes': 0, 'Martes': 0, 'Miercoles': 0, 'Jueves': 0, 'Viernes': 0}
]

with open('Data/semana01.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fields = ['Nombre', 'Domingo','Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes']
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    
    for p in people:
        writer.writerow(p)
    

# Extraer el vector de 128 elementos
# Codificar los rostros extraidos

imageFacesPath = 'C:/Users/2109g/Desktop/Materias/TOPICOS/FaceRecognition/faces'
facesEncodings = []
facesNames = []


for file_name in os.listdir(imageFacesPath):
    image = cv2.imread(imageFacesPath + '/' + file_name)
    # Face_recognition utilizara Dlib y Dlib necesita las imagenes en RGB y no en BGR
    # Por defecto, OpenCV lee las imagenes en BGR, asi que hacemos la transformacion
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Codificamos cada una de las imagenes
    f_coding = face_recognition.face_encodings(image, known_face_locations=[(0, 150, 150, 0)])[0] # Arriba, derecha, abajo, izquierda
    facesEncodings.append(f_coding)
    facesNames.append(file_name.split('.')[0]) # El nombre de los archivos jpg se separa por '.'

print(facesEncodings)
print(facesNames)


###################################################################
# LEYENDO VIDEO

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Detector facial
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

attendance_list = []

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.flip(frame, 1)
    orig = frame.copy()
    faces = faceClassif.detectMultiScale(frame, 1.1, 5)

    for (x, y, w, h) in faces:
        face = orig[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        actual_face_encoding = face_recognition.face_encodings(face, known_face_locations=[(0, w, h, 0)])[0]
        result = face_recognition.compare_faces(facesEncodings, actual_face_encoding) # Se compara el rostro guardado con el rostro actual con el que queremos comparar
        print(result)

        if True in result:
            index = result.index(True)
            name = facesNames[index]
            color = (125, 220, 0)
            #print(name)
            if name not in attendance_list:
                attendance_list.append(name)
        else:
            name = 'Unknown'
            color = (50, 50, 255)

        if nombre_dia == 'Sunday':
            set_attendance('Domingo', attendance_list)

        cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y + h + 25), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Frame', frame)
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

