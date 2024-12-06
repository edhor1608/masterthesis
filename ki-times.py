import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Original data from the image for KI levels and positions
data = {
    "KI": [120, 110, 100, 90, 80, 70],
    "P1": [69.099, 70.763, 72.673, 74.682, 77.293, 81.629],
    "P5": [69.444, 71.264, 73.002, 75.038, 78.521, 82.458],
    "P10": [69.568, 71.575, 73.405, 75.975, 79.499, 83.525],
    "P20": [70.618, 72.483, 74.235, 77.234, 80.936, 85.501],
}

# Create a DataFrame from the data
df_original = pd.DataFrame(data)

# Set KI as the index for easier interpolation
df_original.set_index("KI", inplace=True)

# Create a new index for all KI levels from 70 to 120
ki_levels_full = np.arange(70, 121)

# Use linear interpolation to fill in the values for all KI levels
df_full = df_original.reindex(ki_levels_full).interpolate(method="linear")

# Round the interpolated lap times to three decimal places

df_full_rounded = df_full.round(3)

# Display the rounded interpolated data for the rain track
print(df_full_rounded)

# Plotting the interpolated lap times for all KI levels
plt.figure(figsize=(12, 8))

# Plot each position's lap times across all KI levels
for position in df_full_rounded.columns:
    plt.plot(df_full_rounded.index, df_full[position], label=f"{position}", marker="o")

plt.xlabel("KI Level")
plt.ylabel("Lap Time (seconds)")
plt.title("Lap Times for Different KI Levels in AMS2")
plt.legend(title="Position")
plt.grid(True)
plt.show()

# New data for a different track and with rain
data_rain = {
    "KI": [120, 110, 100, 90, 80, 70],
    "P1": [54.144, 55.607, 57.006, 58.981, 61.719, 64.78],
    "P5": [54.455, 55.861, 57.374, 59.341, 62.148, 65.216],
    "P10": [54.88, 56.166, 57.83, 59.864, 62.79, 66.294],
    "P20": [55.568, 57.577, 58.906, 61.396, 64.688, 68.428],
}

# Create a DataFrame from the new data
df_rain = pd.DataFrame(data_rain)

# Set KI as the index for easier interpolation
df_rain.set_index("KI", inplace=True)

# Create a new index for all KI levels from 70 to 120
ki_levels_full_rain = np.arange(70, 121)

# Use linear interpolation to fill in the values for all KI levels
df_full_rain = df_rain.reindex(ki_levels_full_rain).interpolate(method="linear")

# Round the interpolated lap times to three decimal places
df_full_rain_rounded = df_full_rain.round(3)

# Display the rounded interpolated data for the rain track
print(df_full_rain_rounded)

# Plotting the interpolated lap times for the rain track
plt.figure(figsize=(12, 8))

# Plot each position's lap times across all KI levels for the rain track
for position in df_full_rain_rounded.columns:
    plt.plot(
        df_full_rain_rounded.index,
        df_full_rain_rounded[position],
        label=f"{position}",
        marker="o",
    )

plt.xlabel("KI Level")
plt.ylabel("Lap Time (seconds)")
plt.title("Lap Times for Different KI Levels in AMS2 (Rain Track)")
plt.legend(title="Position")
plt.grid(True)
plt.show()
