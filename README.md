# Projekt i implementacja systemu SI detekcji oznaczeń pojazdów mechanicznych

Projekt stworzony w ramach mojej pracy inżynierskiej. Jest to inteligentny system do zarządzania parkingiem, który w czasie rzeczywistym wykrywa samochody, odczytuje ich tablice rejestracyjne i zapisuje zdarzenia do bazy danych. Całość działa lokalnie na Raspberry Pi.

## Technologie
- Język: Python 3
- Computer Vision & AI: OpenCV, EasyOCR, ONNX Runtime (modele YOLO)
- Baza danych: SQLite
- Sprzęt: Raspberry Pi 4 + Kamera CSI + Wyświetlacz 1.8TFT SPI

##  Główne funkcje
- Wykrywanie tablic rejestracyjnych w kadrze z kamery za pomocą lekkiego modelu ONNX.
- Odczytywanie tekstu z tablic rejestracyjnych za pomocą EasyOCR.
- Automatyczne rejestrowanie wjazdów (data, godzina, numer rejestracyjny) do lokalnej bazy SQLite.
- Wyświetlanie podglądu z kamery na żywo z nałożonymi ramkami (bounding boxes).

##  Demo
- Brak, system działa na komputerze, jestem w trakcie implementacji na raspberry

##  Jak to odpalić u siebie?

Jeśli chcesz uruchomić ten projekt na swoim Raspberry Pi lub komputerze z Linuxem:

1. Sklonuj repozytorium lub pobierz pliki.
2. Upewnij się, że masz podłączoną kamerkę USB.
3. Zainstaluj potrzebne biblioteki wpisując w terminalu:
   ```bash
   sudo apt update
   sudo apt install tesseract-ocr -y
