import cv2
import onnxruntime as ort
import numpy as np
import easyocr
import sqlite3
import datetime
import re
import time

# ==========================================
# 1. LOGIKA BAZY DANYCH
# ==========================================
def check_vehicle(plate):
    # Łączymy się z Twoim wygenerowanym plikiem bazy
    conn = sqlite3.connect('bazaParking.db')
    c = conn.cursor()
    c.execute("SELECT status FROM pojazdy WHERE rejestracja=?", (plate,))
    res = c.fetchone()
    conn.close()
    
    # Skoro w Twojej bazie testowej mamy tylko status (1 - wjazd, 0 - zakaz),
    # dostosowałem logikę do tych danych.
    if res:
        status = res[0]
        if status == 1:
            return {"akcja": "WJAZD", "kolor": (0, 255, 0)} # Zielony (BGR)
        else:
            return {"akcja": "ZAKAZ", "kolor": (0, 0, 255)} # Czerwony
            
    return {"akcja": "NIEZNANY", "kolor": (0, 165, 255)} # Pomarańczowy

# ==========================================
# 2. RYSOWANIE INTERFEJSU (Ekran 128x160)
# ==========================================
def draw_ui(kolor_tla, tekst_glowny, tekst_pomocniczy=""):
    # Tworzymy pusty obraz o proporcjach ekranu 128x160 (odwrócone dla OpenCV: wys x szer)
    ui = np.zeros((128, 160, 3), dtype=np.uint8)
    ui[:] = kolor_tla
    
    # Rysujemy tekst na środku
    cv2.putText(ui, tekst_glowny, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    if tekst_pomocniczy:
        cv2.putText(ui, tekst_pomocniczy, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Powiększamy okno na ekranie monitora, żeby było w ogóle coś widać podczas testów
    ui_display = cv2.resize(ui, (480, 384), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('WYSWIETLACZ 128x160', ui_display)
    cv2.waitKey(1)

# ==========================================
# 3. GŁÓWNY SYSTEM WIZYJNY
# ==========================================
print("[INFO] Ładowanie potężnego modelu ONNX (to potrwa chwilę)...")
session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

print("[INFO] Ładowanie silnika EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

print("[INFO] Start kamery...")
cap = cv2.VideoCapture(0)
# Wymuszamy na kamerze minimalny bufor (ochrona przed zawieszeniem Raspberry Pi)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_counter = 0

# Ekran startowy (Szary)
draw_ui((150, 150, 150), "GOTOWY", "Czekam...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
        
    frame_counter += 1
    
    # OMINIĘCIE KLATEK: Przepuszczamy tylko co 10 klatkę do ciężkiego modelu
    if frame_counter % 10 != 0:
        # Nasłuchujemy wcisnięcia 'q' żeby móc bezpiecznie wyjść
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
        
    h_orig, w_orig = frame.shape[:2]
    
    # Przygotowanie obrazu pod ONNX
    img = cv2.resize(frame, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    # Detekcja YOLO ONNX
    outputs = session.run(None, {input_name: img})
    preds = outputs[0][0]
    
    mask = preds[:, 4] > 0.4
    filtered_preds = preds[mask]
    
    boxes = []
    scores = []
    
    for p in filtered_preds:
        cls_conf = p[5]
        score = p[4] * cls_conf
        if score > 0.4:
            cx, cy, w, h = p[0], p[1], p[2], p[3]
            x1 = int((cx - w / 2) * (w_orig / 640))
            y1 = int((cy - h / 2) * (h_orig / 640))
            bw = int(w * (w_orig / 640))
            bh = int(h * (h_orig / 640))
            boxes.append([x1, y1, bw, bh])
            scores.append(float(score))

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.4)

    # Jeśli system znalazł ramkę z rejestracją
    if len(indices) > 0:
        draw_ui((0, 255, 255), "ANALIZA...", "Prosze czekac") # Żółty ekran
        
        for i in indices.flatten():
            bx, by, bw, bh = boxes[i]
            x1, y1 = max(0, bx), max(0, by)
            x2, y2 = min(w_orig, bx + bw), min(h_orig, by + bh)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            # Przygotowanie wycinka pod EasyOCR
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            
            # Odczyt sztuczną inteligencją
            ocr_res = reader.readtext(thresh, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            if ocr_res:
                raw_text = "".join(ocr_res).upper()
                text = re.sub(r'[^A-Z0-9]', '', raw_text)
                
                # Jeśli odczytało co najmniej 4 znaki
                if len(text) > 3:
                    # 1. Sprawdzamy w bazie
                    dane = check_vehicle(text)
                    
                    # 2. Rysujemy odpowiedni kolor i tekst na ekranie
                    draw_ui(dane["kolor"], dane["akcja"], text)
                    
                    # 3. Zapisujemy log w konsoli
                    print(f"[WYNIK] Rozpoznano: {text} -> Status: {dane['akcja']}")
                    
                    # 4. Zatrzymujemy działanie na 3 sekundy, żeby wczytać komunikat
                    time.sleep(3)
                    
                    # 5. Opróżniamy bufor kamery z przestarzałych klatek
                    for _ in range(5): cap.read()
                    
                    # 6. Wracamy do szarego ekranu czuwania
                    draw_ui((150, 150, 150), "GOTOWY", "Czekam...")
                    break # Wychodzimy z pętli żeby nie czytać tej samej blachy kilka razy

    # Nacisnij 'q' na klawiaturze, aby bezpiecznie wyłączyć program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Zamykanie systemu...")
        break

cap.release()
cv2.destroyAllWindows()