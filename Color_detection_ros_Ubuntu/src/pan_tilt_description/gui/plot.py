#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# 1. Konfiguracja ścieżek do katalogów
DATA_DIR = "/workspace/src/pan_tilt_description/data"
FIGURES_DIR = os.path.join(DATA_DIR, "Figures")

# Automatyczne tworzenie folderu Figures, jeśli jeszcze nie istnieje
os.makedirs(FIGURES_DIR, exist_ok=True)

# 2. Wyszukiwanie wszystkich plików CSV w podanym folderze
pliki_csv = glob.glob(os.path.join(DATA_DIR, "*.csv"))

if not pliki_csv:
    print(f"BŁĄD: Nie znaleziono żadnych plików CSV w folderze:\n{DATA_DIR}")
    exit(1)

print(f"Znaleziono {len(pliki_csv)} plików CSV. Rozpoczynam generowanie wykresów...\n")

# 3. Pętla przetwarzająca każdy znaleziony plik
for plik in pliki_csv:
    try:
        df = pd.read_csv(plik)
        
        # Pobieranie danych z nagłówków
        regulator_type = df['Regulator'].iloc[0] if 'Regulator' in df.columns else 'Nieznany'
        p1_pan = df['Param1_Pan'].iloc[0] if 'Param1_Pan' in df.columns else 0
        p2_pan = df['Param2_Pan'].iloc[0] if 'Param2_Pan' in df.columns else 0
        p1_tilt = df['Param1_Tilt'].iloc[0] if 'Param1_Tilt' in df.columns else 0
        p2_tilt = df['Param2_Tilt'].iloc[0] if 'Param2_Tilt' in df.columns else 0
        
        # Ekstrakcja nazwy pliku do tytułu wykresu (bez całej ścieżki)
        nazwa_bazowa = os.path.splitext(os.path.basename(plik))[0]
        
        if regulator_type == "PID":
            title_text = f'Odpowiedź Skokowa Systemu Nadążnego - Regulator PID z Gain Scheduling\n[{nazwa_bazowa}] PAN (Kp: {p1_pan}, Kd: {p2_pan}) | TILT (Kp: {p1_tilt}, Kd: {p2_tilt})'
        elif regulator_type == "BANG_BANG":
            title_text = f'Odpowiedź Skokowa Systemu Nadążnego - Regulator Trójpozycyjny (Bang-Bang)\n[{nazwa_bazowa}] Prędkość kroku: PAN: {p1_pan} | TILT: {p1_tilt}'
        else:
            title_text = f'Odpowiedź Skokowa Systemu Nadążnego\n[{nazwa_bazowa}]'

        czas = df['Czas_s'].to_numpy()
        uchyb_x = df['Uchyb_X_px'].to_numpy()
        uchyb_y = df['Uchyb_Y_px'].to_numpy()
        
        if 'Deadband' in df.columns:
            deadband = df['Deadband'].to_numpy()
        else:
            deadband = np.full(len(czas), 25.0)

        # Inicjalizacja obszaru roboczego
        plt.figure(figsize=(12, 9))
        
        # --- Oś PAN ---
        plt.subplot(2, 1, 1)
        plt.plot(czas, uchyb_x, label='Uchyb osi PAN', color='#2980b9', linewidth=2)
        plt.axhline(y=0, color='#e74c3c', linestyle='-', linewidth=1.5, label='Środek tarczy (Cel)')
        plt.fill_between(czas, deadband, -deadband, color='#2ecc71', alpha=0.2, label='Dynamiczna Strefa Nieczułości')
        plt.title(title_text, fontsize=12, fontweight='bold')
        plt.ylabel('Uchyb PAN [px]', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        max_uchyb = max(np.max(np.abs(uchyb_x)), np.max(np.abs(uchyb_y)), 100)
        plt.ylim(-max_uchyb * 1.1, max_uchyb * 1.1)
        
        # --- Oś TILT ---
        plt.subplot(2, 1, 2)
        plt.plot(czas, uchyb_y, label='Uchyb osi TILT', color='#f39c12', linewidth=2)
        plt.axhline(y=0, color='#e74c3c', linestyle='-', linewidth=1.5, label='Środek tarczy (Cel)')
        plt.fill_between(czas, deadband, -deadband, color='#2ecc71', alpha=0.2, label='Dynamiczna Strefa Nieczułości')
        plt.xlabel('Czas [s]', fontsize=12)
        plt.ylabel('Uchyb TILT [px]', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(-max_uchyb * 1.1, max_uchyb * 1.1)
        
        plt.tight_layout()
        
        # --- ZAPIS DO FOLDERU FIGURES ---
        out_filename = f'wykres_{nazwa_bazowa}.png'
        out_path = os.path.join(FIGURES_DIR, out_filename)
        
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[SUKCES] Zapisano wykres: Figures/{out_filename}")
        
        # BARDZO WAŻNE: Zamykamy wykres po zapisie, aby zwolnić RAM!
        plt.close()

    except Exception as e:
        print(f"[BŁĄD] Nie udało się przetworzyć pliku {nazwa_bazowa}: {e}")

print(f"\nZakończono pracę! Sprawdź folder:\n{FIGURES_DIR}")