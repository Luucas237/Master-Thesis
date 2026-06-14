#!/usr/bin/env python3
import pandas as pd
import numpy as np

df = pd.read_csv("/workspace/src/pan_tilt_description/data/GOOD_BB_circural_80_cm.csv")

rmse_x = np.sqrt(np.mean(df['Uchyb_X_px']**2))
rmse_y = np.sqrt(np.mean(df['Uchyb_Y_px']**2))

max_error_x = df['Uchyb_X_px'].abs().max()

print(f"RMSE X: {rmse_x:.2f}")
print(f"Maksymalny błąd dynamiczny X: {max_error_x} px")