import os
import json
import glob
from detect_yolov8 import detect_objects_yolov8

"""
Modul principal pentru gestionarea detecției de obiecte și căutarea lor în imagini.
Șterge rezultatele vechi, rulează detecția pe imagini și permite căutarea unui obiect după nume.


Resurse folosite:
- https://github.com/ultralytics/ultralytics/tree/main
- https://github.com/opencv/opencv-python
- https://docs.ultralytics.com/modes/predict/
- https://chat.openai.com/
"""

IMAGE_DIR = "images"
OUTPUT_JSON = "results.json"

# Șterge fișierul JSON de rezultate, dacă există deja
if os.path.exists("results.json"):
    os.remove("results.json")
    print("[INFO] Fișierul results.json a fost șters.")

# Șterge toate imaginile de output generate anterior
for file in glob.glob("output_*.jpg") + glob.glob("output_*.png"):
    os.remove(file)
    print(f"[INFO] Imagine ștearsă: {file}")

def main():
    """
    Rulează procesul principal: detectează obiecte în imagini și permite căutarea unui obiect dorit.
    """
    # Dicționar cu traduceri RO -> EN pentru nume de obiecte
    traduceri = {
        "minge": ["sports ball", "baseball glove"],
        "om": ["person"],
        "masina": ["car"],
        "bicicleta": ["bicycle"],
        "caine": ["dog"],
        "scaun": ["chair"]
    }

    # Rulează detecția obiectelor dacă rezultatele nu există deja
    if not os.path.exists(OUTPUT_JSON):
        detect_objects_yolov8(IMAGE_DIR, OUTPUT_JSON)

    # Primește obiectul de căutat de la utilizator
    obiect_input = input("Introduceți obiectul căutat: ").strip().lower()
    # Alege clasele în engleză asociate obiectului introdus
    clase_de_cautat = traduceri.get(obiect_input, [obiect_input])

    # Încarcă rezultatele detecției
    with open(OUTPUT_JSON, "r") as f:
        data = json.load(f)

    # Caută imaginile în care apare obiectul specificat
    imagini_gasite = [
        entry["image"] for entry in data
        if any(obj.lower() in map(str.lower, entry["objects"]) for obj in clase_de_cautat)
    ]

    # Afișează rezultatul căutării
    if imagini_gasite:
        print(f"\nObiectul '{obiect_input}' a fost găsit în imaginile:")
        for img in imagini_gasite:
            print(f" - {img}")
    else:
        print(f"\nObiectul '{obiect_input}' NU a fost găsit în nicio imagine.")

if __name__ == "__main__":
    main()
