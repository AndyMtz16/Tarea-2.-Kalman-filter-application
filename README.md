# Tarea-2.-Kalman-filter-application
YOLO Cats Tracking 🐾

Este proyecto implementa detección y seguimiento de objetos (gatos) en vídeo utilizando YOLOv8 y el algoritmo SORT. Además, dibuja las trayectorias de movimiento de cada objeto y muestra el progreso de procesamiento usando la libreria Trackers

*Descripción*

Detección de objetos: Se utiliza ultralytics con el modelo yolov8m.pt para detectar gatos en cada frame.

Tracking: Se aplica el algoritmo SORT (Simple Online and Realtime Tracking) a las detecciones para asignar un ID único a cada gato y mantener su rastro.

Trayectorias: Se guardan los centros de cada bounding box por frame y se dibujan líneas verdes conectando los puntos, mostrando el recorrido de cada gato.

*Desarrollo*

Para la implementación del seguimiento de las trayectorias de los objetos (en esta caso gatos) se utiliza una libreria llamada Trackers. Está se publico recientemente por lo que aún no existe mucha documentación al respecto. Dependiendo de tu procesador es que tan rapido analizara el video. En caso de contar con un procesador de NVIDIA, se recomienda utilizar cuda para el procesamiento de YOLO. De lo contrario, solo hay que tomar en cuenta que pueda tomar un poco de tiempo en completar el analísis aunque sea un video corto. 

Como se menciona anteriormente, este código utiliza el modelo de YOLO v8. Se puede utilizar otros modelos de Ultralytrics. En este caso se decidio por el v8 debido al tipo de etiquetas con el que cuenta actualmente de forma nativa. 
