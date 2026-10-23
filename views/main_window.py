"""CustomTkinter view: widget construction and presentation only."""
import customtkinter as ctk


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Safety Helmet Detection System")
        self.geometry("1200x760")
        self.minsize(900, 600)
        self.on_select_image = None
        self.on_detect_image = None
        self.on_save_result = None
        self.on_reset = None
        self.on_select_video = None
        self.on_toggle_webcam = None
        self._original_pil = None
        self._result_pil = None
        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        header = ctk.CTkFrame(self, corner_radius=12)
        header.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        ctk.CTkLabel(header, text="SAFETY HELMET DETECTION", font=("Segoe UI", 24, "bold")).pack(anchor="w", padx=20, pady=(15, 4))
        ctk.CTkLabel(header, text="Computer Vision • YOLO Object Detection", text_color="gray").pack(anchor="w", padx=20, pady=(0, 15))

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, padx=20, pady=(0, 15), sticky="nsew")
        content.grid_columnconfigure(0, weight=4, uniform="panels")
        content.grid_columnconfigure(1, weight=5, uniform="panels")
        content.grid_rowconfigure(0, weight=3)
        content.grid_rowconfigure(1, weight=2)

        # Scrollable controls prevent buttons from being clipped on smaller windows.
        self.controls = ctk.CTkScrollableFrame(content, corner_radius=12, label_text="Pengaturan Deteksi", label_font=("Segoe UI", 18, "bold"))
        self.controls.grid(row=0, column=0, rowspan=2, padx=(0, 10), sticky="nsew")
        self.controls.grid_columnconfigure(0, weight=1)

        self.image_button = ctk.CTkButton(self.controls, text="Pilih Gambar", command=self._select_image, height=40)
        self.image_button.grid(row=0, column=0, padx=18, pady=(12, 6), sticky="ew")
        self.video_button = ctk.CTkButton(self.controls, text="Pilih Video & Deteksi", command=self._select_video, height=40)
        self.video_button.grid(row=1, column=0, padx=18, pady=6, sticky="ew")
        self.webcam_button = ctk.CTkButton(self.controls, text="Mulai Webcam Live", command=self._toggle_webcam, height=40)
        self.webcam_button.grid(row=2, column=0, padx=18, pady=6, sticky="ew")
        # Simpan warna asli tombol agar bisa dikembalikan setelah mode "Stop" (merah).
        self._webcam_idle_colors = (self.webcam_button.cget("fg_color"), self.webcam_button.cget("hover_color"))
        self.file_label = ctk.CTkLabel(self.controls, text="Belum ada gambar dipilih", wraplength=320, text_color="gray", anchor="w")
        self.file_label.grid(row=3, column=0, padx=18, pady=8, sticky="ew")
        ctk.CTkLabel(self.controls, text="Confidence Threshold", anchor="w").grid(row=4, column=0, padx=18, pady=(18, 5), sticky="ew")
        self.confidence_slider = ctk.CTkSlider(self.controls, from_=0.10, to=0.90, number_of_steps=16, command=self._update_confidence)
        self.confidence_slider.set(0.70)
        self.confidence_slider.grid(row=5, column=0, padx=18, pady=5, sticky="ew")
        self.confidence_label = ctk.CTkLabel(self.controls, text="Confidence: 0.70", anchor="w")
        self.confidence_label.grid(row=6, column=0, padx=18, pady=5, sticky="ew")
        self.detect_button = ctk.CTkButton(self.controls, text="Jalankan Deteksi", command=self._detect_image, height=42)
        self.detect_button.grid(row=7, column=0, padx=18, pady=(20, 8), sticky="ew")
        self.save_button = ctk.CTkButton(self.controls, text="Simpan Hasil", command=self._save_result, state="disabled")
        self.save_button.grid(row=8, column=0, padx=18, pady=6, sticky="ew")
        self.reset_button = ctk.CTkButton(self.controls, text="Reset", command=self._reset, fg_color="gray35", hover_color="gray25")
        self.reset_button.grid(row=9, column=0, padx=18, pady=6, sticky="ew")
        self.model_label = ctk.CTkLabel(self.controls, text="Model: memeriksa...", text_color="gray", wraplength=320, anchor="w", justify="left")
        self.model_label.grid(row=10, column=0, padx=18, pady=(20, 16), sticky="ew")

        preview_panel = ctk.CTkFrame(content, corner_radius=12)
        preview_panel.grid(row=0, column=1, padx=(0, 0), pady=(0, 8), sticky="nsew")
        preview_panel.grid_columnconfigure((0, 1), weight=1)
        preview_panel.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(preview_panel, text="Pratinjau", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, columnspan=2, padx=15, pady=12)
        self.original_preview = ctk.CTkLabel(preview_panel, text="Gambar asli", text_color="gray")
        self.original_preview.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")
        self.result_preview = ctk.CTkLabel(preview_panel, text="Hasil deteksi", text_color="gray")
        self.result_preview.grid(row=1, column=1, padx=8, pady=8, sticky="nsew")

        summary = ctk.CTkFrame(content, corner_radius=12)
        summary.grid(row=1, column=1, pady=(2, 0), sticky="nsew")
        self.status_label = ctk.CTkLabel(summary, text="Status: Siap", font=("Segoe UI", 14, "bold"), anchor="w")
        self.status_label.pack(anchor="w", padx=18, pady=(12, 5))
        self.result_label = ctk.CTkLabel(summary, text="Jumlah objek: —", anchor="w", justify="left")
        self.result_label.pack(anchor="w", padx=18, pady=4)
        self.time_label = ctk.CTkLabel(summary, text="Waktu inferensi: —", anchor="w")
        self.time_label.pack(anchor="w", padx=18, pady=4)
        self.video_progress = ctk.CTkProgressBar(summary)
        self.video_progress.set(0)
        self.video_progress.pack(fill="x", padx=18, pady=(8, 3))
        self.progress_label = ctk.CTkLabel(summary, text="Progres video: belum dimulai", text_color="gray", anchor="w")
        self.progress_label.pack(anchor="w", padx=18, pady=(0, 4))
        self.review_label = ctk.CTkLabel(summary, text="Peringatan: belum ada hasil untuk ditinjau", text_color="orange", wraplength=500, anchor="w", justify="left")
        self.review_label.pack(anchor="w", padx=18, pady=(4, 12))

    # Button bridges: the view does not contain application logic.
    def _select_image(self):
        if self.on_select_image: self.on_select_image()
    def _select_video(self):
        if self.on_select_video: self.on_select_video()
    def _toggle_webcam(self):
        if self.on_toggle_webcam: self.on_toggle_webcam()
    def _detect_image(self):
        if self.on_detect_image: self.on_detect_image()
    def _save_result(self):
        if self.on_save_result: self.on_save_result()
    def _reset(self):
        if self.on_reset: self.on_reset()
    def _update_confidence(self, value):
        self.confidence_label.configure(text=f"Confidence: {float(value):.2f}")

    def get_confidence(self): return float(self.confidence_slider.get())
    def set_model_status(self, text): self.model_label.configure(text=text)
    def set_status(self, text): self.status_label.configure(text=text)
    def set_file_label(self, text): self.file_label.configure(text=text)
    def set_result_text(self, text): self.result_label.configure(text=text)
    def set_time_text(self, text): self.time_label.configure(text=text)
    def set_review_text(self, text): self.review_label.configure(text=text)
    def enable_save(self, enabled): self.save_button.configure(state="normal" if enabled else "disabled")
    def set_detect_enabled(self, enabled): self.detect_button.configure(state="normal" if enabled else "disabled")
    def set_video_controls_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.video_button.configure(state=state)
        self.webcam_button.configure(state=state)
        self.detect_button.configure(state=state)
        self.image_button.configure(state=state)
        self.reset_button.configure(state=state)
    def set_webcam_running(self, running):
        """Atur tampilan saat webcam aktif/nonaktif. Tombol webcam sendiri tetap bisa diklik (jadi Stop)."""
        state = "disabled" if running else "normal"
        self.image_button.configure(state=state)
        self.video_button.configure(state=state)
        self.detect_button.configure(state=state)
        self.reset_button.configure(state=state)
        if running:
            self.webcam_button.configure(text="Stop Webcam", fg_color="#c0392b", hover_color="#962d22")
            self.enable_save(False)
            self.original_preview.configure(image=None, text="Webcam live")
            self.original_preview.image = None
            self.result_preview.configure(image=None, text="Menyiapkan kamera...")
            self.result_preview.image = None
            self._original_pil = None
            self._result_pil = None
        else:
            idle_fg, idle_hover = self._webcam_idle_colors
            self.webcam_button.configure(text="Mulai Webcam Live", fg_color=idle_fg, hover_color=idle_hover)
            self.original_preview.configure(image=None, text="Gambar asli")
            self.original_preview.image = None
    def set_live_frame(self, image):
        """Tampilkan frame webcam (PIL) di panel hasil deteksi."""
        self._show_pil_image(image, self.result_preview)
    def set_progress(self, value, text):
        self.video_progress.set(max(0, min(1, float(value))))
        self.progress_label.configure(text=text)
    def _show_pil_image(self, image, widget):
        preview = image.copy()
        preview.thumbnail((420, 320))
        display = ctk.CTkImage(light_image=preview, dark_image=preview, size=preview.size)
        widget.configure(image=display, text="")
        widget.image = display
    def set_original_image(self, image):
        self._original_pil = image
        self._show_pil_image(image, self.original_preview)
    def set_result_image(self, image):
        self._result_pil = image
        self._show_pil_image(image, self.result_preview)
    def clear_result_preview(self):
        self._result_pil = None
        self.result_preview.configure(image=None, text="Belum dideteksi")
        self.result_preview.image = None
    def reset_view(self):
        self._original_pil = None
        self._result_pil = None
        for widget, text in ((self.original_preview, "Gambar asli"), (self.result_preview, "Hasil deteksi")):
            widget.configure(image=None, text=text)
            widget.image = None
        self.file_label.configure(text="Belum ada gambar dipilih")
        self.set_status("Status: Siap")
        self.set_result_text("Jumlah objek: —")
        self.set_time_text("Waktu inferensi: —")
        self.set_review_text("Peringatan: belum ada hasil untuk ditinjau")
        self.enable_save(False)
        self.set_progress(0, "Progres video: belum dimulai")
