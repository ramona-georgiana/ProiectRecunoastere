import os
import json
from ultralytics import YOLO

"""
Modul pentru detecția automată a obiectelor în imagini folosind YOLOv8.
"""

def detect_objects_yolov8(image_dir, output_json):
    """
    Detectează obiectele specificate în toate imaginile dintr-un director,
    salvează rezultatele în fișier JSON și generează imagini cu bounding box-uri.
    """
    # Încarcă modelul YOLOv8 pre-antrenat
    model = YOLO("yolov8s.pt")  # model precis

    results_data = []  # Stochează rezultatele pentru toate imaginile

    # Parcurge toate fișierele din directorul de imagini
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(image_dir, filename)

            # Aplică modelul pe imagine cu prag de încredere 0.25
            results = model(path, conf=0.25)

            detected_objects = []  # Stochează clasele detectate pentru această imagine
            for r in results:
                for c in r.boxes.cls:
                    class_name = model.names[int(c)]
                    detected_objects.append(class_name)

            # Afișează în consolă ce obiecte au fost detectate (pentru debugging)
            print(f"[DEBUG] {filename} -> {detected_objects}")

            # Salvează imaginea cu bounding box-urile trasate
            results[0].save(filename=f"output_{filename}")

            # Adaugă rezultatele pentru această imagine în lista finală
            results_data.append({
                "image": filename,
                "objects": detected_objects
            })

    # Scrie rezultatele tuturor imaginilor în fișierul JSON de ieșire
    with open(output_json, "w") as f:
        json.dump(results_data, f, indent=4)

    # Anunță finalizarea procesului
    print(f"[INFO] Detectare completă. Rezultatele au fost salvate în {output_json}")
