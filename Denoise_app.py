import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import time
import os
import traceback
from settings.settings import PATHS,NAMES
import socket

# Server settings (must match server.py)
HOST = '127.0.0.1'
PORT = 65432

# --- Configuration ---
RECORD_DURATION_SECONDS = 5
DEFAULT_SAMPLE_RATE = 16000
CHANNELS = 1 # Explicitly define channel count

AUDIO_INPUT_FOLDER = PATHS['input_directory'] # Define the folder name
AUDIO_OUTPUT_FOLDER = PATHS['output_directory']

input_filename = NAMES['input_file']
output_filename = NAMES['output_file']

input_file_path = os.path.join(AUDIO_INPUT_FOLDER, input_filename)
output_file_path = os.path.join(AUDIO_OUTPUT_FOLDER, output_filename)

# --- External Denoise sever  ---
def denoise_audio_placeholder():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print("Connected to denoise server.")
            s.sendall(b'denoise')

            # Wait for the server to finish
            data = s.recv(1024)

        if data == b'done':
            print("Denoising finished successfully!")
            return True
        else:
            print("Denoising failed or unknown server response.")
            return False

    except Exception as e:
        print(f"Failed to connect to server or run denoising: {e}")
        messagebox.showerror("Failed to connect",
                             f"Failed to connect to server or run denoising: {e}")
        return False

def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x+ 50}+{y - 50}")

def _get_button_fg_color():
    """Gets the default fg_color for buttons based on theme."""
    try:
        return ctk.ThemeManager.theme["CTkButton"]["fg_color"]
    except Exception:
        return ("#3B8ED0", "#1F6AA5")

# --- Main Application Class ---
class AudioDenoiseApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Audio Denoiser")
        #self.geometry("900x730")
        center_window(self, 900, 750)

        self.protocol("WM_DELETE_WINDOW", self.cleanup_and_exit)

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        print("-" * 20 + " Audio Devices " + "-" * 20)
        try:
            print(sd.query_devices())
            # Try to get default input device info more reliably
            default_input_idx = sd.default.device[0] # Get default input device index
            if default_input_idx == -1:
                 print("\nNo default input device configured in sounddevice/system.")
            else:
                default_input_device_info = sd.query_devices(device=default_input_idx)
                print(f"\nSelected Default Input Device (Index {default_input_idx}):\n{default_input_device_info}")

            # Check if *any* input device exists
            input_devices = sd.query_devices(kind='input')
            if not input_devices:
                 print("\nNo input devices found by sounddevice.")
                 messagebox.showwarning("Audio Input Error", "No microphone/input device found. Recording will not be possible.")

        except Exception as e:
            print(f"\nCould not query audio devices: {e}")
            messagebox.showerror("Audio Device Error", f"Could not query audio devices.\nMake sure sounddevice and its dependencies (like PortAudio) are installed correctly.\nError: {e}")
        print("-" * 55)


        self.original_audio = None
        self.original_sr = None
        self.original_filename = None
        self.denoised_audio = None
        self.denoised_sr = None
        self.is_recording = False
        self.recording_thread = None
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.playing_source = None # 'original' or 'denoised'
        self.playback_thread = None

        # Get the default button color for resetting later
        self.default_button_fg_color = _get_button_fg_color()
        self.default_button_hover_color = "#105e9c"
        self.stop_button_fg_color = "#DB3E3E"
        self.stop_button_hover_color = "#C73434"

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1) # Plot area gets extra space
        self.grid_rowconfigure((0, 2, 3), weight=0) # Frames take minimum space

        # --- Top Frame ---
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.grid(row=0, column=0, padx=20, pady=(5, 5), sticky="ew")
        self.top_frame.grid_columnconfigure((0, 1, 2), weight=1) # Make buttons expand equally

        button_pady_top = 10
        button_padx_top = 15

        self.record_button = ctk.CTkButton(self.top_frame, text="🎤 Record Audio", command=self.toggle_recording ,hover_color=self.default_button_hover_color)
        self.record_button.grid(row=0, column=0, padx=button_padx_top, pady=button_pady_top, sticky="ew")

        self.upload_button = ctk.CTkButton(self.top_frame, text="📁 Upload Audio", command=self.upload_audio, hover_color=self.default_button_hover_color)
        self.upload_button.grid(row=0, column=1, padx=button_padx_top, pady=button_pady_top, sticky="ew")

        self.play_orig_button = ctk.CTkButton(self.top_frame, text="▶️ Play Original", command=lambda: self.toggle_playback('original'), state="disabled" ,hover_color=self.default_button_hover_color)
        self.play_orig_button.grid(row=0, column=2, padx=button_padx_top, pady=button_pady_top, sticky="ew")

        # --- Plot Frame ---
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=1, column=0, padx=20, pady=5, sticky="nsew")
        self.plot_frame.grid_rowconfigure((0, 1), weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)

        # Get theme colors for matplotlib
        matplotlib_bg_color = self._get_matplotlib_rgb("CTkFrame", "fg_color")
        plot_text_color = self._get_matplotlib_rgb("CTkLabel", "text_color")


        plot_common_kwargs = {"facecolor": matplotlib_bg_color}
        tick_common_kwargs = {"axis": 'both',"colors": plot_text_color}
        spine_common_kwargs = {"color": plot_text_color}
        title_common_kwargs = {"color": plot_text_color}

        # Original plot
        self.fig_orig, self.ax_orig = plt.subplots()
        self.fig_orig.patch.set(**plot_common_kwargs)
        self.ax_orig.set(**plot_common_kwargs)
        self.ax_orig.tick_params(**tick_common_kwargs)
        for spine in self.ax_orig.spines.values(): spine.set(**spine_common_kwargs)
        self.ax_orig.set_title("Original Audio Waveform", **title_common_kwargs)
        self.canvas_orig = FigureCanvasTkAgg(self.fig_orig, master=self.plot_frame)
        self.canvas_orig_widget = self.canvas_orig.get_tk_widget()
        self.canvas_orig_widget.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.fig_orig.tight_layout()

        # Denoised plot
        self.fig_denoised, self.ax_denoised = plt.subplots()
        self.fig_denoised.patch.set(**plot_common_kwargs)
        self.ax_denoised.set(**plot_common_kwargs)
        self.ax_denoised.tick_params(**tick_common_kwargs)
        for spine in self.ax_denoised.spines.values(): spine.set(**spine_common_kwargs)
        self.ax_denoised.set_title("Denoised Audio Waveform", **title_common_kwargs)
        self.canvas_denoised = FigureCanvasTkAgg(self.fig_denoised, master=self.plot_frame)
        self.canvas_denoised_widget = self.canvas_denoised.get_tk_widget()
        self.canvas_denoised_widget.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.fig_denoised.tight_layout()

        # --- Controls Frame ---
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        self.controls_frame.grid_columnconfigure((0, 1, 2), weight=1) # Make buttons expand equally

        button_pady_controls = 10
        button_padx_controls = 15

        self.denoise_button = ctk.CTkButton(self.controls_frame, text="✨ Denoise", command=self.denoise_audio, state="disabled" ,hover_color=self.default_button_hover_color)
        self.denoise_button.grid(row=0, column=0, padx=button_padx_controls, pady=button_pady_controls, sticky="ew")

        self.save_button = ctk.CTkButton(self.controls_frame, text="💾 Save Denoised", command=self.save_denoised_audio,state="disabled", hover_color=self.default_button_hover_color)
        self.save_button.grid(row=0, column=1, padx=button_padx_controls, pady=button_pady_controls, sticky="ew")

        self.play_denoised_button = ctk.CTkButton(self.controls_frame, text="▶️ Play Denoised", command=lambda: self.toggle_playback('denoised'), state="disabled" , hover_color=self.default_button_hover_color)
        self.play_denoised_button.grid(row=0, column=2, padx=button_padx_controls, pady=button_pady_controls, sticky="ew")

        # --- Status Frame ---
        self.status_frame = ctk.CTkFrame(self, fg_color="transparent") # Transparent background
        self.status_frame.grid(row=3, column=0, padx=20, pady=(5, 5), sticky="ew")
        self.status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(self.status_frame, text="Status: Ready", anchor="w")
        self.status_label.grid(row=0, column=0, padx=0, pady=0, sticky="ew")

        # --- Initial State ---
        self.update_plot(self.ax_orig, self.canvas_orig, None, None, "Original Audio Waveform")
        self.update_plot(self.ax_denoised, self.canvas_denoised, None, None, "Denoised Audio Waveform")
        self.after(100, self.check_audio_queue) # Start checking the queue for recording results
        self.enable_buttons_after_processing() # Set initial button states

    def cleanup_and_exit(self):
        """Handles cleanup before exiting the application."""
        print("Cleaning up and exiting...")
        # Stop any ongoing recording or playback
        self.is_recording = False
        self.is_playing = False
        try:
            sd.stop()  # Stop any active audio streams
            sd.default.reset()  # Reset sounddevice state
            print("Stopped and reset sounddevice streams.")
        except Exception as e:
            print(f"Error stopping/resetting sounddevice: {e}")

        # Cancel scheduled after callbacks
        if self.queue_check_id is not None:
            try:
                self.after_cancel(self.queue_check_id)
                print("Canceled check_audio_queue callback.")
            except Exception as e:
                print(f"Error canceling check_audio_queue callback: {e}")

        # Close any wait popup
        try:
            self._close_wait_popup()
            print("Closed wait popup.")
        except Exception as e:
            print(f"Error closing wait popup: {e}")

        # Close matplotlib figures
        try:
            if hasattr(self, 'fig_orig') and self.fig_orig is not None:
                plt.close(self.fig_orig)
                print("Closed original figure.")
            if hasattr(self, 'fig_denoised') and self.fig_denoised is not None:
                plt.close(self.fig_denoised)
                print("Closed denoised figure.")
        except Exception as e:
            print(f"Error closing matplotlib figures: {e}")

        # Destroy all Tkinter widgets
        try:
            # Destroy child widgets first
            for widget in self.winfo_children():
                try:
                    widget.destroy()
                    print(f"Destroyed widget: {widget}")
                except Exception as e:
                    print(f"Error destroying widget {widget}: {e}")
            if self.winfo_exists():
                self.destroy()
                print("Destroyed main Tkinter window.")
        except Exception as e:
            print(f"Error destroying Tkinter window: {e}")

        # Ensure customtkinter cleans up
        try:
            ctk.set_appearance_mode("System")  # Reset appearance mode to default
            print("Reset customtkinter appearance mode.")
        except Exception as e:
            print(f"Error resetting customtkinter: {e}")

        # Exit the application
        print("Exiting application.")
        self.quit()  # Stop the Tkinter mainloop
        sys.exit(0)  # Forcefully exit the Python process


    def _get_matplotlib_rgb(self, theme_element, theme_key):
        """Convert CTk theme color string (or tuple) to matplotlib RGB float tuple."""
        try:
            color_tuple_or_str = ctk.ThemeManager.theme[theme_element][theme_key]
            # Use _apply_appearance_mode to get the currently active color string
            color_string = self._apply_appearance_mode(color_tuple_or_str)
            # Convert the hex string to RGB tuple (0-65535 range)
            rgb_16bit = self.winfo_rgb(color_string)
            # Convert to matplotlib's expected float format (0.0-1.0 range)
            rgb_float = tuple(c / 65535.0 for c in rgb_16bit)
            return rgb_float
        except Exception as e:
            print(f"[ColorError] Error converting color {theme_element}.{theme_key}: {e}")
            # Return black as a fallback
            return 0.0, 0.0, 0.0

    def update_status(self, message):
        self.status_label.configure(text=f"Status: {message}")

    def save_input_audio(self):
        if self.original_audio is None or self.original_sr is None:
            messagebox.showwarning("No Audio", "There is no input audio to save.")
            return

        default_filename = NAMES['input_file']
        file_path = os.path.join(AUDIO_INPUT_FOLDER, default_filename)

        try:
            # Clip audio data to valid range [-1.0, 1.0]
            audio_to_save = np.clip(self.original_audio, -1.0, 1.0)

            if os.path.exists(file_path):
                os.remove(file_path)

            # Save the audio file (will create a new file)
            sf.write(file_path, audio_to_save, self.original_sr)

        except Exception as e:
            messagebox.showerror("Input Error", f"Could not save input audio file:\n{file_path}\n\nError: {e}")

    def _stop_any_playback(self):
        """Stops any ongoing playback and resets its state immediately."""
        if self.is_playing:
            print("Stopping current playback...")
            sd.stop()
            # Force immediate state reset, thread's finally might be delayed
            self._reset_playback_state(self.playing_source, stopped_by_action=True)

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if self.is_recording: return

        self._stop_any_playback()

        messagebox.showwarning(
            "Microphone Check",
            "Please make sure your microphone is available and turned ON.\n"
            "If it is OFF, please enable it in your device settings to record audio properly.\n\n"
            "Click OK to start recording."
        )

        # --- Check for *any* input device first ---
        try:
            input_devices = sd.query_devices(kind='input')
            if not input_devices:
                 messagebox.showerror("Microphone Error", "No microphone/input device found.\nCannot start recording.")
                 return
            # Check if a *default* device is specifically set
            default_input_idx = sd.default.device[0]
            if default_input_idx == -1:
                 messagebox.showwarning("Microphone Warning", "No default input device is set.\nSounddevice will try to pick one, but it might not be the one you want.")
                 # We can still proceed, sd.InputStream might pick a working device
            else:
                 print(f"Attempting to use default input device index: {default_input_idx}")

        except Exception as e:
            print(f"[AudioError] Error querying input devices before recording: {e}")
            messagebox.showerror("Device Query Error", f"Could not check for microphones:\n{e}")
            return
        # --- End Input Device Check ---

        # --- Check selected device capabilities ---
        try:
            # This will use the default input device if 'device' is not specified
            sd.check_input_settings(samplerate=DEFAULT_SAMPLE_RATE, channels=CHANNELS)
            print(f"Default input device appears suitable for {DEFAULT_SAMPLE_RATE} Hz, {CHANNELS} Channel(s).")
        except Exception as e:
            print(f"[AudioError] Error checking input device settings: {e}")
            messagebox.showerror("Recording Setup Error",
                                 f"The selected/default input device doesn't support the required settings ({DEFAULT_SAMPLE_RATE}Hz, {CHANNELS}ch) or could not be opened.\n\n"
                                 f"Error details: {e}\n\n"
                                 "Please check your system's audio settings and ensure the correct microphone is selected and working.")
            return
        # --- End Capability Check ---


        self.is_recording = True
        self.original_filename = "recording" # Default name for recordings
        self.record_button.configure(text="⏹️ Stop Recording", fg_color=self.stop_button_fg_color ,hover_color=self.stop_button_hover_color)
        self.upload_button.configure(state="disabled")
        # Disable other potentially interfering buttons
        self.enable_buttons_after_processing(recording=True)

        self.update_status(f"Recording for {RECORD_DURATION_SECONDS}s...")

        # Clear previous audio data and plots
        self.original_audio = None
        self.original_sr = None
        self.denoised_audio = None
        self.denoised_sr = None
        self.update_plot(self.ax_orig, self.canvas_orig, None, None, "Original Audio Waveform")
        self.update_plot(self.ax_denoised, self.canvas_denoised, None, None, "Denoised Audio Waveform")

        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record_audio_thread, args=(DEFAULT_SAMPLE_RATE, RECORD_DURATION_SECONDS))
        self.recording_thread.daemon = True
        self.recording_thread.start()

    def stop_recording(self):
        if not self.is_recording: return
        self.is_recording = False # Signal the thread to stop
        sd.stop()
        self.update_status("Stopping recording...")
        # The thread will finish, put data (or None) in the queue, and check_audio_queue will handle UI updates


    def _record_audio_thread(self, samplerate, duration):
        recorded_chunks = []
        stream_error = None
        try:
            # Callback function to collect audio data
            def callback(indata, frames, time, status):
                if status:
                    # Report issues like buffer overflows
                    print(f"[AudioCallbackStatus] {status}")
                if self.is_recording:
                    recorded_chunks.append(indata.copy())
                else:
                    # Stop requested via the flag
                    raise sd.CallbackStop

            # Use InputStream context manager for recording
            with sd.InputStream(samplerate=samplerate, channels=CHANNELS, dtype='float32', callback=callback):
                # Wait for the specified duration or until is_recording becomes False
                for _ in range(int(duration * 10)): # Check every 100ms
                    if not self.is_recording:
                        break
                    time.sleep(0.1)

            # After the stream closes (or was stopped)
            if recorded_chunks:
                audio_data = np.concatenate(recorded_chunks, axis=0)
                self.audio_queue.put((audio_data, samplerate))
            else:
                # No data recorded (e.g., stopped immediately or error before callback ran)
                 if not self.is_recording: # Check if the stop was intentional
                     print("Recording stopped before any data was captured.")
                     self.audio_queue.put((None, None)) # Indicate intentional stop, no data

        except sd.CallbackStop:
            # Normal stop via a flag leading to CallbackStop raise
            if recorded_chunks:
                 audio_data = np.concatenate(recorded_chunks, axis=0)
                 self.audio_queue.put((audio_data, samplerate))
            else:
                 print("Recording stopped via callback.")
                 self.audio_queue.put((None, None)) # Indicate intentional stop, no data

        except Exception as e:
            stream_error = e
            # Log the full traceback for detailed debugging
            print(f"[AudioError] Exception during recording stream: {e}\n{traceback.format_exc()}")
            self.audio_queue.put((None, None)) # Signal error state

        finally:
             # If an error occurred within the stream context, show it
             if stream_error:
                  self.after(0, lambda err=stream_error: messagebox.showerror("Recording Error",
                                                           f"An error occurred during recording:\n{err}\n\n"
                                                           "Check console for details. Ensure microphone is connected, not in use by another app, and permissions are granted."))
             # Ensure is_recording is false if the thread exits unexpectedly
             self.is_recording = False


    def check_audio_queue(self):
        """Checks the queue for results from the recording thread."""
        try:
            # Get data non-blockingly
            audio_data, samplerate = self.audio_queue.get_nowait()
            # Recording has finished or was stopped/failed
            self.is_recording = False
            self.record_button.configure(text="🎤 Record Audio", fg_color=self.default_button_fg_color, hover_color=self.default_button_hover_color)
            self.upload_button.configure(state="normal")

            if audio_data is not None and samplerate is not None:
                self.update_status("Recording complete.")
                self.original_audio = audio_data
                self.original_sr = samplerate
                self.denoised_audio = None # Clear any old denoised result
                self.denoised_sr = None
                self.save_input_audio()
                self.update_plot(self.ax_orig, self.canvas_orig, self.original_audio, self.original_sr, "Original Audio Waveform")
                self.update_plot(self.ax_denoised, self.canvas_denoised, None, None, "Denoised Audio Waveform")
            else:
                # Recording stopped with no data or failed
                self.update_status("Recording stopped or failed.")
                if self.original_audio is None:
                    self.update_plot(self.ax_orig, self.canvas_orig, None, None, "Original Audio Waveform")
                    self.update_plot(self.ax_denoised, self.canvas_denoised, None, None, "Denoised Audio Waveform")

            # Re-enable buttons based on the current state
            self.enable_buttons_after_processing()

        except queue.Empty:
            # Queue is empty, recording is still ongoing or nothing new
            pass
        finally:
            # Reschedule the check only if the window still exists
            if self.winfo_exists():
                self.after(100, self.check_audio_queue)

    def upload_audio(self):
        if self.is_recording:
             messagebox.showwarning("Busy", "Please stop recording before uploading.")
             return
        self._stop_any_playback() # Stop playback before loading
        file_path = filedialog.askopenfilename(
            title="Select an Audio File",
            filetypes=(("Audio Files", "*.wav *.ogg"), ("All Files", "*.*"))
        )
        if not file_path:
            self.update_status("File selection cancelled.")
            return

        self.update_status(f"Loading {os.path.basename(file_path)}...")
        self.enable_buttons_after_processing(processing=True) # Disable buttons during load
        try:
            # Clear the previous state before loading
            self.original_audio = None
            self.original_sr = None
            self.denoised_audio = None
            self.denoised_sr = None
            self.original_filename = os.path.splitext(os.path.basename(file_path))[0]

            # Read an audio file using soundfile
            audio_data, samplerate = sf.read(file_path, dtype='float32')

            # Convert to mono if stereo by averaging channels
            if audio_data.ndim > 1:
                self.update_status(f"Converting stereo to mono...")
                audio_data = np.mean(audio_data, axis=1)

            self.original_audio = audio_data
            self.original_sr = samplerate

            self.update_status("Audio loaded successfully.")
            self.save_input_audio()
            # Update plots with new data
            self.update_plot(self.ax_orig, self.canvas_orig, self.original_audio, self.original_sr, "Original Audio Waveform")
            self.update_plot(self.ax_denoised, self.canvas_denoised, None, None, "Denoised Audio Waveform")

        except Exception as e:
            self.update_status(f"Error loading file: {e}")
            messagebox.showerror("File Load Error", f"Could not load or process the audio file:\n{file_path}\n\nError: {e}")
            # Reset state on error
            self.original_audio = None
            self.original_sr = None
            self.original_filename = None
            self.update_plot(self.ax_orig, self.canvas_orig, None, None, "Original Audio Waveform")
            self.update_plot(self.ax_denoised, self.canvas_denoised, None, None, "Denoised Audio Waveform")

        finally:
            # Re-enable buttons after a loading attempt (success or fail)
            self.enable_buttons_after_processing()

    def update_plot(self, ax, canvas, audio_data, sr, title):
        """Updates a specified matplotlib axis and redraws the canvas."""
        ax.clear() # Clear previous plot elements

        # Reapply styling for consistency after clear()
        matplotlib_bg_color = self._get_matplotlib_rgb("CTkFrame", "fg_color")
        plot_text_color = self._get_matplotlib_rgb("CTkLabel", "text_color")
        plot_line_color = self._get_matplotlib_rgb("CTkButton", "fg_color")
        grid_color = self._get_matplotlib_rgb("CTkFrame", "border_color")

        ax.set_facecolor(matplotlib_bg_color)
        ax.tick_params(axis='both', colors=plot_text_color)
        for spine in ax.spines.values(): spine.set_color(plot_text_color)
        ax.set_title(title, color=plot_text_color)
        ax.set_xlabel("Time [s]", color=plot_text_color)
        ax.set_ylabel("Amplitude", color=plot_text_color)
        ax.grid(True, linestyle='--', alpha=0.6, color=grid_color) # Add grid

        # Plot audio data if available
        if audio_data is not None and sr is not None and len(audio_data) > 0:
            # Create time axis
            time_axis = np.linspace(0, len(audio_data) / sr, num=len(audio_data))
            # Plot the waveform
            ax.plot(time_axis, audio_data, color=plot_line_color, linewidth=0.8)
            # Adjust plot limits for better visualization
            min_val, max_val = audio_data.min(), audio_data.max()
            padding = (max_val - min_val) * 0.1 + 1e-6 # Add small epsilon for zero amplitude
            ax.set_ylim(min_val - padding, max_val + padding)
            ax.set_xlim(0, time_axis[-1]) # Set x-limit to audio duration
        else:
            # Set default limits if no data
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        # Redraw the canvas
        try:
            # Use tight_layout to prevent labels overlapping
            ax.figure.tight_layout(pad=2) # Adjust padding as needed
        except ValueError:
             # Can sometimes happen with empty plots, ignore
             pass
        finally:
             canvas.draw()


    def toggle_playback(self, source):
        """Handles clicks on Play Original/Denoised buttons."""
        if self.is_recording:
            messagebox.showwarning("Busy", "Cannot play audio while recording.")
            return

        audio_data = self.original_audio if source == 'original' else self.denoised_audio
        sr = self.original_sr if source == 'original' else self.denoised_sr
        button = self.play_orig_button if source == 'original' else self.play_denoised_button

        if audio_data is None or sr is None:
            self.update_status(f"No {'original' if source == 'original' else 'denoised'} audio to play.")
            return

        if self.is_playing:
            # If the *same* source is clicked again, stop it
            if self.playing_source == source:
                self.update_status("Stopping playback...")
                sd.stop()
            else:
                self.update_status("Stopping previous playback...")
                sd.stop()
                # Schedule the new playback to start shortly after stopping the old one
        else:
            # No playback is active, start the requested one
            self._start_playback(source)


    def _start_playback(self, source):
        """Internal function to initiate audio playback."""
        if self.is_playing: # Should not happen if logic in toggle_playback is correct, but safety check
             print("Warning: _start_playback called while already playing.")
             sd.stop()
             self._reset_playback_state(self.playing_source, stopped_by_action=True)

        audio_data = self.original_audio if source == 'original' else self.denoised_audio
        sr = self.original_sr if source == 'original' else self.denoised_sr
        button = self.play_orig_button if source == 'original' else self.play_denoised_button
        other_button = self.play_denoised_button if source == 'original' else self.play_orig_button

        if audio_data is None or sr is None: # Should be checked before calling, but re-check
            return

        self.is_playing = True
        self.playing_source = source
        self.update_status(f"Playing {'original' if source == 'original' else 'denoised'} audio...")

        # Update button appearance
        button.configure(text="⏹️ Stop Playing", fg_color=self.stop_button_fg_color,hover_color=self.stop_button_hover_color)
        # Disable the *other* play button while one is playing
        other_button.configure(state="disabled")
        self.record_button.configure(state="disabled")
        self.upload_button.configure(state="disabled")
        self.denoise_button.configure(state="disabled")
        self.save_button.configure(state="disabled")


        # Create and start the playback thread
        self.playback_thread = threading.Thread(target=self._play_audio_thread, args=(audio_data, sr, source))
        self.playback_thread.daemon = True
        self.playback_thread.start()

    def _play_audio_thread(self, audio_data, sr, source):
        """Thread target for playing audio using sounddevice."""
        play_error = None
        try:
            sd.play(audio_data, sr)
            sd.wait()
            self.is_playing = True
        except Exception as e:
            play_error = e
            print(f"[AudioError] Error during playback: {e}")
            # Schedule error display in main thread
            self.after(0, lambda err=e: self.update_status(f"Playback error: {err}"))
        finally:
            self.after(0, self._reset_playback_state, source)

    def _reset_playback_state(self, source_that_finished, stopped_by_action=False):
        """Resets playback state (flags, button text) in the main thread."""
        # Only reset if the source matches the currently playing one
        if self.is_playing and self.playing_source == source_that_finished:
            self.is_playing = False
            self.playing_source = None
            self.playback_thread = None # Clear thread reference

            button = self.play_orig_button if source_that_finished == 'original' else self.play_denoised_button
            button_text = "▶️ Play Original" if source_that_finished == 'original' else "▶️ Play Denoised"
            button.configure(text=button_text, fg_color=self.default_button_fg_color, hover_color=self.default_button_hover_color)

            # Re-enable buttons based on audio data availability
            self.enable_buttons_after_processing()

            if not stopped_by_action:
                 self.update_status("Playback finished.")
            else:
                 self.update_status("Playback stopped.")


    def _show_wait_popup(self):
        """Creates and shows the 'Please Wait' popup."""
        # Check and destroy existing popup more safely
        existing_popup = getattr(self, 'wait_popup', None)
        if existing_popup is not None:
            print("Existing popup found, attempting to destroy...")
            try:
                if existing_popup.winfo_exists():
                    existing_popup.destroy()
                    print("Existing popup destroyed.")
                else:
                    print("Existing popup did not exist (winfo_exists() returned False).")
            except tk.TclError as e:
                print(f"TclError destroying existing popup (likely already closed): {e}")
            except Exception as e:
                print(f"Unexpected error destroying existing popup: {e}")
            finally:
                self.wait_popup = None
                print("self.wait_popup reset to None after attempting to destroy existing.")

        self.wait_popup = ctk.CTkToplevel(self)
        self.wait_popup.title("Denoising...") # No title bar text
        self.wait_popup.geometry("300x100")
        self.wait_popup.resizable(False, False)
        self.wait_popup.attributes("-topmost", True) # Keep on top

        try:
            main_win_x = self.winfo_x()
            main_win_y = self.winfo_y()
            main_win_width = self.winfo_width()
            main_win_height = self.winfo_height()
            popup_width = 300
            popup_height = 100
            pos_x = main_win_x + (main_win_width // 2) - (popup_width // 2)
            pos_y = main_win_y + (main_win_height // 2) - (popup_height // 2)
            self.wait_popup.geometry(f"{popup_width}x{popup_height}+{pos_x}+{pos_y}")
        except tk.TclError as e:
            print(f"Error centering popup (main window might not be ready?): {e}")
            self.wait_popup.geometry("300x100+200+200")

        self.wait_popup.transient(self)
        # Add label
        wait_label = ctk.CTkLabel(self.wait_popup, text="Please wait...\nDenoising audio", font=ctk.CTkFont(size=14))
        wait_label.pack(expand=True, padx=20, pady=20)
        # Prevent closing via the 'X' button
        self.wait_popup.protocol("WM_DELETE_WINDOW", lambda: None)
        self.wait_popup.lift() # Ensure it's raised above other windows

    def _close_wait_popup(self):
        """Closes the 'Please Wait' popup if it exists."""
        popup_to_close = getattr(self, 'wait_popup', None)
        if popup_to_close is not None:
            try:
                if popup_to_close.winfo_exists():
                    popup_to_close.destroy()
                else:
                    print("Wait popup did not exist when trying to close (winfo_exists() returned False).")
            except tk.TclError as e:
                 print(f"TclError closing popup (likely already closed): {e}")
            except Exception as e:
                 print(f"Unexpected error closing popup: {e}")
            finally:
                 self.wait_popup = None
                 print("self.wait_popup reset to None after close attempt.")
        else:
            print("No wait popup found to close (self.wait_popup was None or missing).")


    def denoise_audio(self):
        if self.original_audio is None or self.original_sr is None:
            self.update_status("Load or record audio first.")
            messagebox.showwarning("Warning", "Please load or record audio before denoising.")
            return
        if self.is_recording:
             messagebox.showwarning("Busy", "Cannot denoise while recording.")
             return

        self._stop_any_playback() # Stop playback before processing

        self.update_status("Denoising...")
        self.enable_buttons_after_processing(processing=True)
        self.record_button.configure(state="disabled")
        self.upload_button.configure(state="disabled")

        # --- Show Wait Popup ---
        self.after(10, self._show_wait_popup)
        # --- End Show Wait Popup ---

        # Start denoising in a background thread
        denoise_thread = threading.Thread(target=self._denoise_thread_job)
        denoise_thread.daemon = True
        denoise_thread.start()

    def _denoise_thread_job(self):
        """Thread target for running the denoising function."""
        try:
            if denoise_audio_placeholder():
                denoised_data, denoised_samplerate = sf.read(output_file_path, dtype='float32')
                # Store results
                self.denoised_audio = denoised_data
                self.denoised_sr = denoised_samplerate
            # Schedule UI update in the main thread
            self.after(0, self.update_denoise_ui)

        except Exception as e:
            # Log error and show a message in the main thread
            print(f"[DenoiseError] Exception during denoise process: {e}\n{traceback.format_exc()}")
            self.after(0, lambda: self.update_status(f"Denoising error: {e}"))
            self.after(0, lambda err=e: messagebox.showerror("Denoise Error", f"An error occurred during denoising:\n{err}"))
            # Ensure buttons are re-enabled even if denoising fails
            self.after(0, self.enable_buttons_after_processing)

        finally:
            self.after_idle(self._close_wait_popup)

    def update_denoise_ui(self):
        """Updates UI elements after denoising is complete (called from the main thread)."""
        if self.denoised_audio is not None:
            self.update_status("Denoising complete.")
            # Update the denoised plot
            self.update_plot(self.ax_denoised, self.canvas_denoised, self.denoised_audio, self.denoised_sr, "Denoised Audio Waveform")
        else:
            self.update_status("Denoising failed or returned no data.")
            # Clear the denoised plot if denoising failed
            self.update_plot(self.ax_denoised, self.canvas_denoised, None, None, "Denoised Audio Waveform")

        # Re-enable buttons based on the new state
        self.enable_buttons_after_processing()

    def enable_buttons_after_processing(self, processing=False, recording=False):
        """Centralized function to enable/disable buttons based on state."""
        # Determine if audio data exists
        orig_exists = self.original_audio is not None
        denoised_exists = self.denoised_audio is not None

        # Conditions for disabling: processing, recording, or audio not ready
        play_orig_disabled = processing or recording or not orig_exists or self.is_playing # Disable if playing anything
        denoise_disabled = processing or recording or not orig_exists or self.is_playing
        play_denoised_disabled = processing or recording or not denoised_exists or self.is_playing
        save_disabled = processing or recording or not denoised_exists or self.is_playing

        # Configure states (don't change text/color here, only state)
        self.play_orig_button.configure(state="disabled" if play_orig_disabled else "normal")
        self.denoise_button.configure(state="disabled" if denoise_disabled else "normal")
        self.play_denoised_button.configure(state="disabled" if play_denoised_disabled else "normal")
        self.save_button.configure(state="disabled" if save_disabled else "normal")
        self.record_button.configure(state="normal")

        # The upload button is disabled only during recording
        self.upload_button.configure(state="disabled" if recording else "normal")

        # Special handling for play buttons if currently playing
        if self.is_playing:
             # Ensure the button *for the currently playing source* remains enabled (to act as stop)
             if self.playing_source == 'original':
                 self.play_orig_button.configure(state="normal")
                 self.play_denoised_button.configure(state="disabled") # Keep other disabled
             elif self.playing_source == 'denoised':
                 self.play_denoised_button.configure(state="normal")
                 self.play_orig_button.configure(state="disabled") # Keep other disabled

    def save_denoised_audio(self):
        if self.denoised_audio is None or self.denoised_sr is None:
            messagebox.showwarning("No Audio", "There is no denoised audio to save.")
            return
        if self.is_playing or self.is_recording:
             messagebox.showwarning("Busy", "Please stop playback or recording before saving.")
             return

        # Suggest a filename based on the original
        default_filename = "denoised_output.wav"
        if self.original_filename:
            default_filename = f"{self.original_filename}_denoised.wav"

        # Ask the user for save location and filename
        file_path = filedialog.asksaveasfilename(
            title="Save Denoised Audio As",
            initialfile=default_filename,
            defaultextension=".wav",
            filetypes=(("WAV files", "*.wav"),
                       ("FLAC files", "*.flac"),
                       ("All Files", "*.*"))
        )

        if not file_path:
            self.update_status("Save cancelled.")
            return

        self.update_status(f"Saving to {os.path.basename(file_path)}...")
        self.enable_buttons_after_processing(processing=True) # Disable buttons during save

        try:
            # Clip audio data to valid range [-1.0, 1.0] before saving
            audio_to_save = np.clip(self.denoised_audio, -1.0, 1.0)

            # Use soundfile to write the audio data
            sf.write(file_path, audio_to_save, self.denoised_sr)

            self.update_status("Denoised audio saved successfully.")
            messagebox.showinfo("Save Successful", f"Denoised audio saved to:\n{file_path}")

        except Exception as e:
            self.update_status(f"Error saving file: {e}")
            messagebox.showerror("Save Error", f"Could not save audio file:\n{file_path}\n\nError: {e}")
        finally:
            # Re-enable buttons after saving attempt
            self.enable_buttons_after_processing()

# --- Main Execution ---
if __name__ == "__main__":
    app = AudioDenoiseApp()
    app.mainloop()