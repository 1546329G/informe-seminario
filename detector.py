import cv2   #libreria para procesar imagenes

# Cargar el clasificador Haar   
# algoritmo utilizado es cascada de haar casde 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Acceder a la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:  #  si no se reconoce el buble se rompe 
        break

    # Convertir a escala de grises para detección
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)     #detecta  rostros de las imegen y devuelve  (x, y, w, h)   de cada rostro 

# Solo dibujar  2 rostro 
#Dibuja rectángulos verdes 
#Muestra un mensaje de que se  detectaron 2 rostros
#También imprime el mensaje en consola de code 

  
    if len(faces) == 2:  # aqui le decimmos que decte 2 rostros
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
  #  ezquina superior son xy
        mensaje = "Exactamente 2 rostros detectados"
        print(mensaje)
        cv2.putText(frame, mensaje, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        # Mostrar mensaj i no hay 2 rostros
        mensaje = f"{len(faces)} rostros detectados - NO EXISTE :)  "
        print(mensaje)
        cv2.putText(frame, mensaje, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

   
   
   
   
   
   
   
   
    # Mostra  frame
    cv2.imshow("Detector - Solo 2 rostros", frame)

    # Salir con 'q' y guardar si había 2 rostros
    if cv2.waitKey(1) & 0xFF == ord('q'):  # tocar q para capt   y espera duante  1 milesegun
        if len(faces) == 2:
            cv2.imwrite("3.jpg", frame)
        break


cap.release()  # creeamos la cmara  

cv2.destroyAllWindows()
