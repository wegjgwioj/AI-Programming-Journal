import pytesseract
from PIL import ImageGrab
import tkinter as tk
from tkinter import messagebox

def capture_screenshot():
    screenshot = ImageGrab.grab()
    screenshot.save("screenshot.png")
    return "screenshot.png"

def ocr_image(image_path):
    text = pytesseract.image_to_string(image_path, lang='eng')
    return text

def show_text():
    image_path = capture_screenshot()
    text = ocr_image(image_path)
    messagebox.showinfo("OCR Result", text)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    show_text()