import matplotlib.pyplot as plt
import numpy as np

pop = np.load("/Users/devincoleman/Downloads/drive-download-20251205T041523Z-3-001/B-97_97.1_pop_fingerprints.npy")      # orange
classic_rock = np.load("/Users/devincoleman/Downloads/drive-download-20251205T041523Z-3-001/The_bayou_95.7_Classic-Rock_fingerprints.npy")  # blue
country = np.load("/Users/devincoleman/Downloads/drive-download-20251205T041523Z-3-001/The_cajun_97.7_Country_fingerprints.npy")           # green
classic_hits = np.load("/Users/devincoleman/Downloads/drive-download-20251205T041523Z-3-001/WRQQ_103.3_Classic_Hits_fingerprints.npy")     # purple

time = np.arange(360) * 0.25  # 360 snapshots × 0.25 s

plt.figure(figsize=(12,6))
plt.plot(time, pop, color='#FF5500', lw=2.2, label='B97.1 – Pop (brickwalled)')
plt.plot(time, classic_rock, color='#0066FF', lw=2.2, label='95.7 The Bayou – Classic Rock')
plt.plot(time, country, color='#00AA00', lw=2.2, label='97.7 The Cajun – Country')
plt.plot(time, classic_hits, color='#AA00AA', lw=2.2, label='103.3 WRQQ – Classic Hits')

plt.axhline(y=0.75, color='gray', linestyle='--', alpha=0.6)
plt.text(75, 0.76, 'Modern pop almost never drops below this line', color='gray', fontsize=11)

plt.title('90-Second Loudness Dynamics of Louisiana FM Stations\n(every dot = 0.25 s frame, normalized RMS energy)', fontsize=14, pad=20)
plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Energy (0–1)')
plt.legend(frameon=True, fancybox=True, shadow=True)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()