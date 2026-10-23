"""Entry point for the Safety Helmet Detection desktop application."""
import customtkinter as ctk
from controllers.app_controller import AppController
from views.main_window import MainWindow


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    view = MainWindow()
    AppController(view)
    view.mainloop()


if __name__ == "__main__":
    main()
