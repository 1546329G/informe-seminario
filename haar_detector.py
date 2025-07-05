import cv2
import os
import urllib.request

def descargar_haar_xml(nombre_archivo, url):
    if not os.path.exists(nombre_archivo):
        print(" Descarando Har Casca.")
        try:
            urllib.request.urlretrieve(url, nombre_archivo)
            print(" Archivo descargado correctamente.")
        except Exception as e:
            print(f" Error al descargar: {e}")
            exit()

def inicializar_camara():
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        print("No se puede acceder a la cámara  :(   ")
        exit()
    return camara

def detectar_rostros(frame, clasificador):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = clasificador.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5)
    return rostros

def dibujar_rostros(frame, rostros):
    for i, (x, y, w, h) in enumerate(rostros):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, f"ID {i + 1}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    conteo_texto = f"Rostros detectados: {len(rostros)}"
    cv2.putText(frame, conteo_texto, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return frame

def main():
    archivo_haar = "haarcascade_frontalface_default.xml"
    url_haar = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml"

    descargar_haar_xml(archivo_haar, url_haar)

    clasificador = cv2.CascadeClassifier(archivo_haar)
    camara = inicializar_camara()

    while True:
        ret, frame = camara.read()
        if not ret:
            break

        rostros = detectar_rostros(frame, clasificador)
        frame = dibujar_rostros(frame, rostros)

        cv2.imshow("Detección de rostros Ha", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if len(rostros) > 0:
                cv2.imwrite("haar_resultado.jpg", frame)
                print(" Imagen haar_resultado.jpg")
            break

    camara.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
