from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.utils import platform
from kivy.graphics.texture import Texture
from kivy.uix.image import Image

import numpy as np
import matplotlib.pyplot as plt
import librosa
import sounddevice as sd
import os
import threading
from datetime import datetime
import io
from PIL import Image as PILImage  #––––––> For decoding PNG to raw bytes

# Android permissions----- NOT TESTED
'''
NOTE-> NOT TESTED YET
'''
SAMPLE_RATE = 44100
RECORD_DURATION = 120
BLOCK_SECONDS = 0.25
HOP_LENGTH = int(SAMPLE_RATE * BLOCK_SECONDS)

class RadioCREEP(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 20
        self.spacing = 15
        self.recording = False
        self.audio_data = []

        Window.clearcolor = (0.08, 0.08, 0.12, 1)

        # Title Label
        self.add_widget(Label(
            text="RadioCREEP\n Music Genre Detector (Beta)",
            markup=True, font_size='22sp', size_hint_y=0.2, halign='center', color= 'blue'
        ))

        self.status = Label(text="Ready • Tap Start", color=(0.9,0.9,1,1), size_hint_y=0.1)
        self.add_widget(self.status)

        self.progress = ProgressBar(max=RECORD_DURATION, size_hint_y=0.05)
        self.add_widget(self.progress)

        btns = BoxLayout(spacing=12, size_hint_y=0.25)
        self.start_btn = Button(text="Start", background_color=(0,0.8,1,1))
        self.stop_btn  = Button(text="Stop & Analyze", disabled=True, background_color=(1,0.4,0.4,1))
        self.plot_btn  = Button(text="View Plots", disabled=True, background_color=(0.2,0.8,0.2,1))
        self.save_btn  = Button(text="Save Data", disabled=True, background_color=(0.8,0.4,1,1))

        for b in (self.start_btn, self.stop_btn, self.plot_btn, self.save_btn):
            b.bind(on_press=self.on_button)
            btns.add_widget(b)
        self.add_widget(btns)

        self.energy_profile = None
        self.avg_fft = None

    def on_button(self, btn):
        {'Start': self.start_recording,
         'Stop & Analyze': self.stop_recording,
         'View Plots': self.show_plots,
         'Save Data': self.save_data}[btn.text]()

    def start_recording(self):
        if self.recording: return
        self.recording = True
        self.audio_data = []
        self.progress.value = 0

        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        self.plot_btn.disabled = self.save_btn.disabled = True
        self.status.text = "Recording..."

        def record():
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=1024) as stream:
                while self.recording:
                    block, _ = stream.read(1024)
                    self.audio_data.append(block)

        threading.Thread(target=record, daemon=True).start()
        Clock.schedule_interval(self.update_progress, 0.2)
        Clock.schedule_once(lambda dt: self.stop_recording() if self.recording else None, RECORD_DURATION)

    def update_progress(self, dt):
        if not self.recording: return False
        secs = len(self.audio_data) * 1024 / SAMPLE_RATE
        self.progress.value = secs
        m, s = divmod(int(secs), 60)
        self.status.text = f"Recording {m:02d}:{s:02d}"
        return True

    def stop_recording(self):
        if not self.recording: return
        self.recording = False
        Clock.unschedule(self.update_progress)
        self.status.text = "Analyzing... please wait...."
        self.stop_btn.disabled = True
        self.start_btn.disabled = False

        threading.Thread(target=self.analyze_in_background, daemon=True).start()

    def analyze_in_background(self):
        if not self.audio_data:
            Clock.schedule_once(lambda dt: self.set_status("No audio captured :("))
            return

        audio_np = np.concatenate(self.audio_data).flatten()

        # Energy profile
        rms = librosa.feature.rms(y=audio_np, frame_length=HOP_LENGTH*4, hop_length=HOP_LENGTH)[0]
        energy = rms / (rms.max() + 1e-12)
        stability = np.std(energy)
        S = np.abs(librosa.stft(audio_np, n_fft=4096, hop_length=HOP_LENGTH))
        freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=4096)
        mask = freqs <= 15000
        self.avg_fft = np.mean(S[mask], axis=1)
        self.energy_profile = energy

        # Classifications
        if stability < 0.13:
            result, color = "MUSIC", "00ff00"
        elif stability > 0.22:
            result, color = "TALK", "ff8800"
        else:
            result, color = "MIXED", "ffff00"

        Clock.schedule_once(lambda dt: self.analysis_complete(result, color, stability))

    def analysis_complete(self, result, color, stability):
        self.status.text = f"{result} • Stability {stability:.3f}"
        self.plot_btn.disabled = False
        self.save_btn.disabled = False
        self.plot_btn.background_color = (0.2, 0.8, 0.2, 1)
        self.save_btn.background_color = (0.8, 0.4, 1, 1)

    def set_status(self, text):
        self.status.text = text

    def show_plots(self):
        if self.energy_profile is None: return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), facecolor='#0c0c0c')
        fig.suptitle('RadioCREEP • Analysis', color='cyan', fontsize=18, fontweight='bold')

        t = np.arange(len(self.energy_profile)) * BLOCK_SECONDS
        ax1.fill_between(t, self.energy_profile, color='#00ffff', alpha=0.7)
        ax1.plot(t, self.energy_profile, color='cyan', lw=2)
        ax1.set_title('Loudness Dynamics', color='white')
        ax1.set_ylabel('Energy')
        ax1.grid(alpha=0.3)
        ax1.set_facecolor('#111')
        ax1.tick_params(colors='white')

        freqs = np.linspace(20, 15000, len(self.avg_fft))
        ax2.semilogy(freqs, self.avg_fft + 1e-8, color='#ff00ff', lw=2)
        ax2.set_title('Spectral Fingerprint', color='white')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.grid(alpha=0.3)
        ax2.set_facecolor('#111')
        ax2.tick_params(colors='white')

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0c0c0c')
        buf.seek(0)
        plt.close(fig)

        img = PILImage.open(buf)
        img = img.convert('RGBA')
        raw_data = img.tobytes()

        tex = Texture.create(size=img.size)
        tex.blit_buffer(raw_data, colorfmt='rgba', bufferfmt='ubyte')
        tex.flip_vertical()

        popup = Popup(title="RadioCREEP • Plots", size_hint=(0.95, 0.95))
        kivy_img = Image(texture=tex)
        close = Button(text="Close", size_hint_y=0.1, background_color=(1,0.3,0.3,1))
        close.bind(on_press=popup.dismiss)
        box = BoxLayout(orientation='vertical')
        box.add_widget(kivy_img)
        box.add_widget(close)
        popup.content = box
        popup.open()

    def save_data(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        np.save(f"RadioCREEP_{ts}.npy", np.column_stack((self.energy_profile, self.avg_fft)))
        self.status(Label( f"Saved RadioCREEP_{ts}.npy", color= 'orange'))


class RadioCREEPApp(App):
    def build(self):
        return RadioCREEP()

if __name__ == '__main__':
    RadioCREEPApp().run()
