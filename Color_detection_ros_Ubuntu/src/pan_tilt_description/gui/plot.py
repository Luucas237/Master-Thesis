#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Wpisz tutaj nazwe swojego nowo wygenerowanego pliku CSV
FILENAME = "/workspace/src/pan_tilt_description/data.csv"

try:
    df = pd.read_csv(FILENAME)
    
    kp_pan = df['Kp_Pan'].iloc[0]
    kd_pan = df['Kd_Pan'].iloc[0]
    kp_tilt = df['Kp_Tilt'].iloc[0]
    kd_tilt = df['Kd_Tilt'].iloc[0]
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(df['Czas_s'].to_numpy(), df['Uchyb_X_px'].to_numpy(), label='Uchyb osi PAN', color='blue', marker='.')
    plt.axhline(y=0, color='r', linestyle='--', label='Punkt docelowy (Środek)')
    plt.axhline(y=25, color='g', linestyle=':', alpha=0.5)
    plt.axhline(y=-25, color='g', linestyle=':', alpha=0.5, label='Martwa Strefa')
    
    plt.title(f'Odpowiedź Skokowa Układu\nPAN (Kp: {kp_pan}, Kd: {kd_pan}) | TILT (Kp: {kp_tilt}, Kd: {kd_tilt})')
    plt.ylabel('Uchyb PAN [px]')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(df['Czas_s'].to_numpy(), df['Uchyb_Y_px'].to_numpy(), label='Uchyb osi TILT', color='orange', marker='.')
    plt.axhline(y=0, color='r', linestyle='--', label='Punkt docelowy (Środek)')
    plt.xlabel('Czas [s]')
    plt.ylabel('Uchyb TILT [px]')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Nie znaleziono pliku {FILENAME}")