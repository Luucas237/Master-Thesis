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


# #!/usr/bin/env python3
# import control as ctrl
# import matplotlib.pyplot as plt
# import numpy as np

# # --- 1. MODEL MATEMATYCZNY OBIEKTU FOPDT ---
# k_obj = 19.0   # Wzmocnienie obiektu
# T_obj = 0.20   # Stała czasowa
# L_obj = 0.15   # Opóźnienie transportowe

# # Aproksymacja Padé 3. rzędu dla opóźnienia transportowego e^(-Ls)
# num_pade, den_pade = ctrl.pade(L_obj, n=3)
# Delay = ctrl.tf(num_pade, den_pade)

# # Transmitancja obiektu inercyjnego G(s) połączona z opóźnieniem
# Plant = ctrl.tf([k_obj], [T_obj, 1]) * Delay

# # --- 2. OSTATECZNE NASTAWY REGULATORA PD (Z Metody Cyklu Granicznego) ---
# Kp = 0.117
# Ki = 0.0       # Brak członu I zapobiega wind-upowi w systemie wizyjnym
# Kd = 0.002
# Tf = 0.05      # Filtr inercyjny dla członu D (50 milisekund)

# # Konstrukcja fizycznie realizowalnego regulatora PD z filtrem
# C_P = ctrl.tf([Kp], [1])
# C_I = ctrl.tf([Ki], [1, 0])
# C_D = ctrl.tf([Kd, 0], [Tf, 1])

# Controller = C_P + C_I + C_D

# # --- 3. ZAMKNIĘCIE PĘTLI SPRZĘŻENIA ZWROTNEGO ---
# # Układ ze sprzężeniem zwrotnym jednostkowym
# System = ctrl.feedback(Controller * Plant, 1)

# # --- 4. SYMULACJA ODPOWIEDZI SKOKOWEJ ---
# # Symulacja przez 3 sekundy ze skokiem w t=0
# time = np.linspace(0, 8, 1000)
# t, y = ctrl.step_response(System, time)

# # --- 5. RYSOWANIE WYKRESU ---
# plt.figure(figsize=(12, 6))
# plt.plot(t, y, label="Odpowiedź układu (PD: Kp=0.117, Kd=0.0071)", color='#27ae60', linewidth=2.5)
# plt.axhline(1.0, color='#e74c3c', linestyle='--', label="Wartość zadana (Cel = 1.0)")

# # Formatowanie wykresu
# plt.title("Symulacja Analitycznych Nastaw Regulatora PD", fontsize=14, fontweight='bold')
# plt.xlabel("Czas [s]", fontsize=12)
# plt.ylabel("Znormalizowana Amplituda", fontsize=12)
# plt.grid(True, which='both', linestyle=':', alpha=0.7)
# plt.legend(loc='lower right')
# plt.show()


#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

# TUTAJ WPISZ NAZWĘ NOWEGO PLIKU CSV
plik_csv = "/workspace/src/pan_tilt_description/data/A_Skok.csv"

print(f"Otwieram plik: {plik_csv}")

try:
    df = pd.read_csv(plik_csv)
    czas = df['Czas_s'].to_numpy()
    uchyb_pan = df['Uchyb_X_px'].to_numpy()

    plt.figure(figsize=(14, 7))
    
    # Rysowanie głównego przebiegu
    plt.plot(czas, uchyb_pan, label='Odpowiedź Skokowa (Otwarta Pętla)', color='#27ae60', linewidth=2.5)
    plt.axhline(y=0, color='#e74c3c', linestyle='-', linewidth=1.5, label='Zero')

    # Opisy i formatowanie
    plt.title(f"Identyfikacja Obiektu - {os.path.basename(plik_csv)}", fontsize=14, fontweight='bold')
    plt.xlabel("Czas [s]", fontsize=12)
    plt.ylabel("Uchyb PAN [px]", fontsize=12)
    
    # Gęsta siatka ułatwiająca odczyt punktów 28.3% i 63.2%
    plt.grid(True, which='major', linestyle='-', alpha=0.8)
    plt.grid(True, which='minor', linestyle='--', alpha=0.4)
    plt.minorticks_on()
    plt.legend(loc='lower right')
    
    plt.show()

except Exception as e:
    print(f"Błąd podczas analizy pliku: {e}")