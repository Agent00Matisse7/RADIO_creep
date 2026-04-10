from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.core.window import Window
import os

import numpy as np
import matplotlib.pyplot as plt
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import librosa
import sounddevice as sd

# Constants
SAMPLE_RATE = 44100
BLOCK_SECONDS = 0.25
print('radioHEAD automation v1.0.1')
print("Please enter time duration in seconds: ")
RECORD_DURATION = int(input()) # changed for 90 sec on 2/2/26
NYQUIST_FREQUENCY = SAMPLE_RATE / 2


class SoundAnalyzer(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 10
        self.spacing = 10

        self.recording = False
        self.energy_profile = None
        self.avg_fft = None
        self.fig = None
        self.elapsed = 0
        self.timer = None

        # This holds the raw audio data
        self.audio_buffer = None

        # UI Elements
        self.status_label = Label(text="Ready to record", size_hint_y=0.1)
        self.add_widget(self.status_label)

        start_btn = Button(text="Start Recording", on_press=self.start_recording, background_color=(0, 1, 0, 1))
        self.add_widget(start_btn)

        stop_btn = Button(text="Stop & Analyze", on_press=self.stop_and_analyze, background_color=(1, 0, 0, 1))
        self.add_widget(stop_btn)

        plot_btn = Button(text="View Plots", on_press=self.show_plots)
        self.add_widget(plot_btn)

        save_btn = Button(text="Save Data (.npy)", on_press=self.save_data)
        self.add_widget(save_btn)

    def start_recording(self, instance):
        if self.recording:
            return

        self.recording = True
        self.elapsed = 0
        self.status_label.text = f"Recording... (0/{RECORD_DURATION}s)"

        try:
            # Pre-allocate the memory for 90 seconds
            self.audio_buffer = np.zeros((RECORD_DURATION * SAMPLE_RATE, 1), dtype='float32')

            # Start recording (non-blocking)
            sd.rec(int(RECORD_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, out=self.audio_buffer)

            self.timer = Clock.schedule_interval(self.update_progress, 1)
            Clock.schedule_once(self.auto_stop, RECORD_DURATION)
        except Exception as e:
            Logger.error(f"Recording error: {e}")
            self.status_label.text = "Mic Error: Check System Permissions"

    def update_progress(self, dt):
        self.elapsed += 1
        self.status_label.text = f"Recording... ({self.elapsed}/{RECORD_DURATION}s)"
        return self.recording

    def auto_stop(self, dt):
        self.stop_and_analyze(None)

    def stop_and_analyze(self, instance):
        if not self.recording:
            return

        self.recording = False
        self.status_label.text = "Analyzing..."
        if self.timer:
            self.timer.cancel()

        try:
            sd.stop()

            # only the portion recorded so far
            actual_samples = int(self.elapsed * SAMPLE_RATE)
            full_audio = self.audio_buffer[:actual_samples].flatten()

            if len(full_audio) < 1000:
                raise ValueError("Recording too short.")

            hop_length = int(SAMPLE_RATE * BLOCK_SECONDS)
            rms = librosa.feature.rms(y=full_audio, frame_length=hop_length * 2, hop_length=hop_length)[0]

            self.energy_profile = rms / np.max(rms) if np.max(rms) > 0 else rms
            stability_score = np.std(self.energy_profile)

            #Prediction logic as of 4/10/2026
            '''
            Stay 'tuned' for more prediction genres.
            '''

            prediction = "MUSIC (Steady)" if stability_score < 0.15 else "Talking/Large Crowds"

            #Compute average FFT
            S = np.abs(librosa.stft(full_audio, n_fft=2048, hop_length=hop_length))
            self.avg_fft = np.mean(S, axis=1)

            self.status_label.text = f"Done, WOOOO! Stability: {stability_score:.4f}\n{prediction}"

        except Exception as e:
            Logger.error(f"Analysis error: {e}")
            self.status_label.text = f"Analysis failed: {str(e)}"

    def show_plots(self, instance):
        if self.energy_profile is None or self.avg_fft is None:
            self.status_label.text = "Record and analyze first!"
            return

        self.fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        time = np.arange(len(self.energy_profile)) * BLOCK_SECONDS
        ax1.plot(time, self.energy_profile, color='orange', label='Energy')
        ax1.set_title('Loudness Dynamics')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Normalized Energy')
        ax1.grid(alpha=0.3)

        frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=2048)
        mask = frequencies <= 15000
        ax2.plot(frequencies[mask], self.avg_fft[mask])
        ax2.set_title('Average FFT Fingerprint')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Purple') #
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        content = BoxLayout(orientation='vertical')
        canvas = FigureCanvasKivyAgg(self.fig)
        content.add_widget(canvas)
        close_btn = Button(text="Close", size_hint_y=0.1)
        content.add_widget(close_btn)

        popup = Popup(title="Analysis Plots", content=content, size_hint=(0.9, 0.9))
        close_btn.bind(on_press=popup.dismiss)
        popup.open()

    def save_data(self, instance):
        if self.energy_profile is None:
            self.status_label.text = "WARNING: Nothing to save Buddy!"
            return
        np.save('analysis_energy.npy', self.energy_profile)
        np.save('analysis_fft.npy', self.avg_fft)
        self.status_label.text = "Data saved to .npy files"
        print(os.path.abspath('analysis_energy.npy'))


class radioHEADApp(App):
    def build(self):
        Window.size = (400, 700)

        return SoundAnalyzer()


if __name__ == '__main__':
    radioHEADApp().run()