import cv2
import os
import urllib.request

def descargar_haar_xml(nombre_archivo, url):    # Verifica si el archivo haarcascade_frontalface_default.xml   #Este archivo contiene el modelo preentrenado para detectar rostro
    if not os.path.exists(nombre_archivo):
        print(" Descarando Har Casca.")
        try:
            urllib.request.urlretrieve(url, nombre_archivo)     # Descarga el archivo desde la URL
            print(" Archivo descargado correctamente.")
        except Exception as e:
            print(f" Error al descargar: {e}")
            exit()

def inicializar_camara():   # Verifica si la cámara está disponible   y si esta lo puede abrir,  si no puede abrirla, el programa se detiene
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        print("No se puede acceder a la cámara  :(   ")
        exit()
    return camara

def detectar_rostros(frame, clasificador):      # Convierte el frame a escala de grises y detecta rostros   
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = clasificador.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5)
    return rostros

def dibujar_rostros(frame, rostros): # Dibuja rectángulos alrededor de los rostros  y muestra esl conteo de rostros en la imagen
    for i, (x, y, w, h) in enumerate(rostros):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Dibuja un rectángulo amarillo alrededor del rostro
        cv2.putText(frame, f"ID {i + 1}", (x, y - 10),                 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # Muestra el ID del rostro

    conteo_texto = f"Rostros detectados: {len(rostros)}"        # Muestra el conteo de rosdtros detectados
    cv2.putText(frame, conteo_texto, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Dibuja el texto en la imagen  
    return frame

def main():         # Función principal que ejecruta el detector de rosstro
    archivo_haar = "haarcascade_frontalface_default.xml"  
    url_haar = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml"      # link del archivo Haar Cascade preentrenado

    descargar_haar_xml(archivo_haar, url_haar)

    clasificador = cv2.CascadeClassifier(archivo_haar)  # Carga el clasificador Haar Cascade para detección de rostros
    camara = inicializar_camara()

    while True:     #  bucle que captura frames de la cámara
        ret, frame = camara.read() 
        if not ret:
            break

        rostros = detectar_rostros(frame, clasificador)  # Detecta rostros en el frame  
        frame = dibujar_rostros(frame, rostros)

        cv2.imshow("Detección de rostros Ha", frame)            # Muestra el frame con los rostros detectados                 

        if cv2.waitKey(1) & 0xFF == ord('q'):   # toma capt con la tecla q 
            if len(rostros) > 0:
                cv2.imwrite("haar_resultado.jpg", frame)
                print(" Imagen haar_resultado.jpg")
            break

    camara.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
