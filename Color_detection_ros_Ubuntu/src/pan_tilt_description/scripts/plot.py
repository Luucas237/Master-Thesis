import pandas as pd
import matplotlib.pyplot as plt

# Wpisz nazwę pliku, który skopiowałeś z Malinki
FILENAME = "/workspace/src/pan_tilt_description/data.csv"

try:
    df = pd.read_csv(FILENAME)
    
    # Pobranie parametrów z pierwszego wiersza
    kp = df['Kp'].iloc[0]
    ki = df['Ki'].iloc[0]
    kd = df['Kd'].iloc[0]
    
    plt.figure(figsize=(10, 8))
    
    # Wykres osi PAN (X)
    plt.subplot(2, 1, 1)
    plt.plot(df['Czas_s'], df['Uchyb_X_px'], label='Uchyb osi PAN', color='blue', marker='.')
    plt.axhline(y=0, color='r', linestyle='--', label='Punkt docelowy (Środek)')
    # Linia oznaczająca martwą strefę (np. 25px)
    plt.axhline(y=25, color='g', linestyle=':', alpha=0.5)
    plt.axhline(y=-25, color='g', linestyle=':', alpha=0.5, label='Martwa Strefa')
    
    plt.title(f'Odpowiedź Skokowa Układu\nKp: {kp}, Ki: {ki}, Kd: {kd}')
    plt.ylabel('Uchyb PAN [px]')
    plt.legend()
    plt.grid(True)
    
    # Wykres osi TILT (Y)
    plt.subplot(2, 1, 2)
    plt.plot(df['Czas_s'], df['Uchyb_Y_px'], label='Uchyb osi TILT', color='orange', marker='.')
    plt.axhline(y=0, color='r', linestyle='--', label='Punkt docelowy (Środek)')
    plt.xlabel('Czas [s]')
    plt.ylabel('Uchyb TILT [px]')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Nie znaleziono pliku {FILENAME}")