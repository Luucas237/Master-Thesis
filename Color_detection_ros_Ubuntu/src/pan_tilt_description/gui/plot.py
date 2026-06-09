#!/usr/bin/env python3
# import control as ctrl
# import matplotlib.pyplot as plt
# import numpy as np

# # --- 1. MODEL MATEMATYCZNY OBIEKTU FOPDT ---
# k_obj = 22.3   # Wzmocnienie obiektu
# T_obj = 0.142   # Stała czasowa
# L_obj = 0.0656   # Opóźnienie transportowe

# # Aproksymacja Padé 3. rzędu dla opóźnienia transportowego e^(-Ls)
# num_pade, den_pade = ctrl.pade(L_obj, n=3)
# Delay = ctrl.tf(num_pade, den_pade)

# # Transmitancja obiektu inercyjnego G(s) połączona z opóźnieniem
# Plant = ctrl.tf([k_obj], [T_obj, 1]) * Delay

# # --- 2. OSTATECZNE NASTAWY REGULATORA PD (Z Metody Cyklu Granicznego) ---
# Kp = 0.058
# Ki = 0.0       # Brak członu I zapobiega wind-upowi w systemie wizyjnym
# Kd = 0.0019
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
# time = np.linspace(0, 2, 1000)
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




# #!/usr/bin/env python3
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import glob

# # 1. Konfiguracja ścieżek do katalogów
# DATA_DIR = "/workspace/src/pan_tilt_description/data"
# FIGURES_DIR = os.path.join(DATA_DIR, "Figures")

# # Automatyczne tworzenie folderu Figures, jeśli jeszcze nie istnieje
# os.makedirs(FIGURES_DIR, exist_ok=True)

# # 2. Wyszukiwanie wszystkich plików CSV w podanym folderze
# pliki_csv = glob.glob(os.path.join(DATA_DIR, "*.csv"))

# if not pliki_csv:
#     print(f"BŁĄD: Nie znaleziono żadnych plików CSV w folderze:\n{DATA_DIR}")
#     exit(1)

# print(f"Znaleziono {len(pliki_csv)} plików CSV. Rozpoczynam generowanie wykresów...\n")

# # 3. Pętla przetwarzająca każdy znaleziony plik
# for plik in pliki_csv:
#     try:
#         df = pd.read_csv(plik)
        
#         # Pobieranie danych z nagłówków (zostawiamy to, by nic nie zepsuć reszty kodu)
#         regulator_type = df['Regulator'].iloc[0] if 'Regulator' in df.columns else 'Nieznany'
#         p1_pan = df['Param1_Pan'].iloc[0] if 'Param1_Pan' in df.columns else 0
#         p2_pan = df['Param2_Pan'].iloc[0] if 'Param2_Pan' in df.columns else 0
#         p1_tilt = df['Param1_Tilt'].iloc[0] if 'Param1_Tilt' in df.columns else 0
#         p2_tilt = df['Param2_Tilt'].iloc[0] if 'Param2_Tilt' in df.columns else 0
        
#         nazwa_bazowa = os.path.splitext(os.path.basename(plik))[0]
#         nazwa_upper = nazwa_bazowa.upper()
        
#         # --- MODYFIKACJA: Sztywno wpisywane tytuły na podstawie nazwy pliku ---
#         if "A_Terrain_PID_lp" in nazwa_upper:
#             title_text = 'Odpowiedź Skokowa Systemu Nadążnego - Regulator PID\nNastawy analityczne: Kp=0.058, Kd=0.0019'
#         elif "A_Terrain_BB_LP" in nazwa_upper:
#             title_text = 'Odpowiedź Skokowa Systemu Nadążnego - Regulator skokowy (Bang-Off-Bang)'
#         else:
#             # Fallback dla innych plików, wzięty z Twojego oryginalnego kodu
#             if regulator_type == "PID":
#                 title_text = f'Odpowiedź Skokowa Systemu Nadążnego - Regulator PID\nNastawy: PAN (Kp: {p1_pan}, Kd: {p2_pan}) | TILT (Kp: {p1_tilt}, Kd: {p2_tilt})'
#             elif regulator_type == "BANG_BANG":
#                 title_text = f'Odpowiedź Skokowa Systemu Nadążnego - Regulator Trójpozycyjny\nPrędkość kroku: PAN: {p1_pan} | TILT: {p1_tilt}'
#             else:
#                 title_text = f'Odpowiedź Skokowa Systemu Nadążnego'
#         # ------------------------------------------------------------------------

#         czas = df['Czas_s'].to_numpy()
#         uchyb_x = df['Uchyb_X_px'].to_numpy()
#         uchyb_y = df['Uchyb_Y_px'].to_numpy()
        
#         if 'Deadband' in df.columns:
#             deadband = df['Deadband'].to_numpy()
#         else:
#             deadband = np.full(len(czas), 25.0)

#         # Inicjalizacja obszaru roboczego
#         plt.figure(figsize=(12, 9))
        
#         # --- Oś PAN ---
#         plt.subplot(2, 1, 1)
#         plt.plot(czas, uchyb_x, label='Uchyb osi PAN', color='#2980b9', linewidth=2)
#         plt.axhline(y=0, color='#e74c3c', linestyle='-', linewidth=1.5, label='Środek tarczy')
#         plt.fill_between(czas, deadband, -deadband, color='#2ecc71', alpha=0.2, label='Dynamiczna Strefa Nieczułości')
#         plt.title(title_text, fontsize=12, fontweight='bold')
#         plt.ylabel('Uchyb PAN [px]', fontsize=12)
#         plt.legend(loc='upper right')
#         plt.grid(True, linestyle='--', alpha=0.7)
        
#         max_uchyb = max(np.max(np.abs(uchyb_x)), np.max(np.abs(uchyb_y)), 100)
#         plt.ylim(-max_uchyb * 1.1, max_uchyb * 1.1)
        
#         # --- Oś TILT ---
#         plt.subplot(2, 1, 2)
#         plt.plot(czas, uchyb_y, label='Uchyb osi TILT', color='#f39c12', linewidth=2)
#         plt.axhline(y=0, color='#e74c3c', linestyle='-', linewidth=1.5, label='Środek tarczy')
#         plt.fill_between(czas, deadband, -deadband, color='#2ecc71', alpha=0.2, label='Dynamiczna Strefa Nieczułości')
#         plt.xlabel('Czas [s]', fontsize=12)
#         plt.ylabel('Uchyb TILT [px]', fontsize=12)
#         plt.legend(loc='upper right')
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.ylim(-max_uchyb * 1.1, max_uchyb * 1.1)
        
#         plt.tight_layout()
        
#         # --- ZAPIS DO FOLDERU FIGURES ---
#         out_filename = f'wykres_{nazwa_bazowa}.png'
#         out_path = os.path.join(FIGURES_DIR, out_filename)
        
#         plt.savefig(out_path, dpi=300, bbox_inches='tight')
#         print(f"[SUKCES] Zapisano wykres: Figures/{out_filename}")
        
#         # BARDZO WAŻNE: Zamykamy wykres po zapisie, aby zwolnić RAM!
#         plt.close()

#     except Exception as e:
#         print(f"[BŁĄD] Nie udało się przetworzyć pliku {nazwa_bazowa}: {e}")

# print(f"\nZakończono pracę! Sprawdź folder:\n{FIGURES_DIR}")


# #!/usr/bin/env python3
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # 1. Konfiguracja ścieżek do katalogów
# DATA_DIR = "/workspace/src/pan_tilt_description/data"
# FIGURES_DIR = os.path.join(DATA_DIR, "Figures")

# # Automatyczne tworzenie folderu Figures, jeśli jeszcze nie istnieje
# os.makedirs(FIGURES_DIR, exist_ok=True)

# # 2. Ścieżka do konkretnego pliku
# plik_csv = os.path.join(DATA_DIR, "A_Terrain_PID_lp.csv")

# # Sprawdzenie, czy plik istnieje
# if not os.path.exists(plik_csv):
#     print(f"[BŁĄD] Nie znaleziono pliku: {plik_csv}")
#     exit(1)

# print(f"Znaleziono plik {plik_csv}. Rozpoczynam generowanie wykresu...")

# try:
#     df = pd.read_csv(plik_csv)
    
#     # --- TYTUŁ NA SZTYWNO ZGODNIE Z WYMOGIEM ---
#     title_text = 'Odpowiedź Skokowa Systemu Nadążnego - Regulator PID\nNastawy: PAN (Kp=0.058, Kd=0.0019) | TILT (Kp=0.058, Kd=0.0019)'
    
#     # -------------------------------------------

#     czas = df['Czas_s'].to_numpy()
#     uchyb_x = df['Uchyb_X_px'].to_numpy()
#     uchyb_y = df['Uchyb_Y_px'].to_numpy()
    
#     if 'Deadband' in df.columns:
#         deadband = df['Deadband'].to_numpy()
#     else:
#         deadband = np.full(len(czas), 25.0)

#     # Inicjalizacja obszaru roboczego
#     plt.figure(figsize=(12, 9))
    
#     # --- Oś PAN ---
#     plt.subplot(2, 1, 1)
#     plt.plot(czas, uchyb_x, label='Uchyb osi PAN', color='#2980b9', linewidth=2)
#     plt.axhline(y=0, color='#e74c3c', linestyle='-', linewidth=1.5, label='Środek tarczy')
#     plt.fill_between(czas, deadband, -deadband, color='#2ecc71', alpha=0.2, label='Dynamiczna Strefa Nieczułości')
#     plt.title(title_text, fontsize=12, fontweight='bold')
#     plt.ylabel('Uchyb PAN [px]', fontsize=12)
#     plt.legend(loc='upper right')
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     # Dynamiczne skalowanie osi Y
#     max_uchyb = max(np.max(np.abs(uchyb_x)), np.max(np.abs(uchyb_y)), 100)
#     plt.ylim(-max_uchyb * 1.1, max_uchyb * 1.1)
    
#     # --- Oś TILT ---
#     plt.subplot(2, 1, 2)
#     plt.plot(czas, uchyb_y, label='Uchyb osi TILT', color='#f39c12', linewidth=2)
#     plt.axhline(y=0, color='#e74c3c', linestyle='-', linewidth=1.5, label='Środek tarczy')
#     plt.fill_between(czas, deadband, -deadband, color='#2ecc71', alpha=0.2, label='Dynamiczna Strefa Nieczułości')
#     plt.xlabel('Czas [s]', fontsize=12)
#     plt.ylabel('Uchyb TILT [px]', fontsize=12)
#     plt.legend(loc='upper right')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.ylim(-max_uchyb * 1.1, max_uchyb * 1.1)
    
#     plt.tight_layout()
    
#     # --- ZAPIS DO FOLDERU FIGURES ---
#     out_filename = 'wykres_A_Terrain_PID_lp.png'
#     out_path = os.path.join(FIGURES_DIR, out_filename)
    
#     plt.savefig(out_path, dpi=300, bbox_inches='tight')
#     print(f"[SUKCES] Zapisano wykres: {out_path}")
    
#     plt.close()

# except Exception as e:
#     print(f"[BŁĄD] Wystąpił problem podczas przetwarzania pliku: {e}")

# #!/usr/bin/env python3
import pandas as pd
import numpy as np

# Wczytanie pliku CSV
df = pd.read_csv("/workspace/src/pan_tilt_description/data/GOOD_BB_circural_80_cm.csv")

# Obliczenie RMSE dla uchybu X i Y
rmse_x = np.sqrt(np.mean(df['Uchyb_X_px']**2))
rmse_y = np.sqrt(np.mean(df['Uchyb_Y_px']**2))

# Obliczenie maksymalnego błędu dynamicznego
max_error_x = df['Uchyb_X_px'].abs().max()

print(f"RMSE X: {rmse_x:.2f}")
print(f"Maksymalny błąd dynamiczny X: {max_error_x} px")