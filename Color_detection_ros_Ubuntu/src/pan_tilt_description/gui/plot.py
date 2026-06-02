#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Wpisz tutaj nazwe swojego nowo wygenerowanego pliku CSV
FILENAME = "/workspace/src/pan_tilt_description/data.csv"

try:
    df = pd.read_csv(FILENAME)
    
    # Pobranie typu regulatora (zabezpieczenie na wypadek starych plików)
    regulator_type = df['Regulator'].iloc[0] if 'Regulator' in df.columns else 'Nieznany'
    
    kp_pan = df['Kp_Pan'].iloc[0]
    kd_pan = df['Kd_Pan'].iloc[0]
    kp_tilt = df['Kp_Tilt'].iloc[0]
    kd_tilt = df['Kd_Tilt'].iloc[0]
    
    # Budowanie profesjonalnego tytułu w zależności od użytego algorytmu
    if regulator_type == "PID":
        title_text = f'Odpowiedź Skokowa Układu Nadążnego - Regulator PID\nPAN (Kp: {kp_pan}, Kd: {kd_pan}) | TILT (Kp: {kp_tilt}, Kd: {kd_tilt})'
    elif regulator_type == "BANG_BANG":
        title_text = f'Odpowiedź Skokowa Układu Nadążnego - Regulator Trójpozycyjny (Bang-Bang)\nStrefa nieczułości: ±25 px'
    else:
        title_text = f'Odpowiedź Skokowa Układu Nadążnego'

    plt.figure(figsize=(10, 8))
    
    # --- Wykres osi PAN ---
    plt.subplot(2, 1, 1)
    # Przywrócone .to_numpy(), aby uniknąć błędu ValueError w Pandas
    plt.plot(df['Czas_s'].to_numpy(), df['Uchyb_X_px'].to_numpy(), label='Uchyb osi PAN', color='#2980b9', linewidth=2)
    
    plt.axhline(y=0, color='#e74c3c', linestyle='-', linewidth=1.5, label='Środek tarczy (Cel)')
    
    # Zacieniowanie strefy nieczułości
    plt.axhspan(-25, 25, color='#2ecc71', alpha=0.2, label='Strefa nieczułości (Deadband)')
    
    plt.title(title_text, fontsize=14, fontweight='bold')
    plt.ylabel('Uchyb PAN [px]', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # --- Wykres osi TILT ---
    plt.subplot(2, 1, 2)
    plt.plot(df['Czas_s'].to_numpy(), df['Uchyb_Y_px'].to_numpy(), label='Uchyb osi TILT', color='#f39c12', linewidth=2)
    
    plt.axhline(y=0, color='#e74c3c', linestyle='-', linewidth=1.5, label='Środek tarczy (Cel)')
    plt.axhspan(-25, 25, color='#2ecc71', alpha=0.2, label='Strefa nieczułości (Deadband)')
    
    plt.xlabel('Czas [s]', fontsize=12)
    plt.ylabel('Uchyb TILT [px]', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Automatyczny zapis do pliku wysokiej jakości (300 dpi) przed wyświetleniem
    out_filename = f'wykres_odpowiedzi_{regulator_type}.png'
    plt.savefig(out_filename, dpi=300)
    print(f"Zapisano wykres jako: {out_filename}")
    
    plt.show()

except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku {FILENAME}. Upewnij się, że wygenerowałeś dane i ścieżka jest poprawna.")
except KeyError as e:
    print(f"BŁĄD: Brak oczekiwanej kolumny w pliku CSV: {e}. Upewnij się, że używasz nowego pliku wygenerowanego po aktualizacji trackera.")