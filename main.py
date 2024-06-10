import tkinter as tk
import customtkinter
import os
import atexit
from tkinter import filedialog
from PIL import Image,ImageTk
import time
import threading
from diffusers import StableDiffusionPipeline
import torch
from torch import autocast
from translate import Translator

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        customtkinter.set_default_color_theme("green")
        self.geometry("1200x700")
        self.title("TVS.AI")
        self.iconbitmap("photo/TVS_LOGO.ico")
        self.resizable(False,False)

        self.switch_var = customtkinter.StringVar(value="off")
        self.switch = customtkinter.CTkSwitch(self, text="Light Mode  ", command=self.switch_event,
                                              variable=self.switch_var,
                                              onvalue="on", offvalue="off")
        self.switch.place(relx=1.0, rely=1.0, anchor=tk.SE)

        self.textbox = customtkinter.CTkTextbox(self, width=450, height=80)
        self.textbox.insert("0.0", "ENTER PROMPT HERE")
        self.textbox.place(relx=0.79, rely=0.25, anchor=tk.CENTER)

        self.button = customtkinter.CTkButton(self, text="GENERATE", command=self.Generate_function)
        self.button.place(relx=0.92, rely=0.35, anchor=tk.CENTER)

        self.textbox.bind("<FocusIn>", self.clear_text)
        self.textbox.bind("<FocusOut>", self.restore_text)
        self.default_text = "ENTER PROMPT HERE"

        self.generate_label = customtkinter.CTkLabel(self, text="Generate", fg_color="transparent",
                                                     font=("Helvetica", 35))
        self.generate_label.place(relx=0.80, rely=0.15, anchor=tk.CENTER)

        self.tvs_label = customtkinter.CTkLabel(self, text="Text Visualization System", fg_color="transparent",
                                                font=("Helvetica", 40))
        self.tvs_label.place(relx=0.5, rely=0.05, anchor=tk.CENTER)

        self.styles_label = customtkinter.CTkLabel(self, text="Styles", fg_color="transparent",
                                                   font=("Helvetica", 35))
        self.styles_label.place(relx=0.24, rely=0.15, anchor=tk.CENTER)

        self.frame1 = customtkinter.CTkFrame(self, width=140, height=140)
        self.frame1.place(relx=0.05, rely=0.20)
        self.image1 = Image.open("photo/CompVis.jpg")
        self.image1 = self.image1.resize((140, 140), Image.Resampling.LANCZOS)
        self.image_tk1 = ImageTk.PhotoImage(self.image1)
        self.label = customtkinter.CTkLabel(self.frame1, image=self.image_tk1, text="")
        self.label.pack()

        self.frame2 = customtkinter.CTkFrame(self, width=140, height=140)
        self.frame2.place(relx=0.20, rely=0.20)
        self.image2 = Image.open("photo/RunwayML.jpg")
        self.image2 = self.image2.resize((140, 140), Image.Resampling.LANCZOS)
        self.image_tk2 = ImageTk.PhotoImage(self.image2)
        self.label = customtkinter.CTkLabel(self.frame2, image=self.image_tk2, text="")
        self.label.pack()

        self.frame3 = customtkinter.CTkFrame(self, width=140, height=140)
        self.frame3.place(relx=0.35, rely=0.20)
        self.image3 = Image.open("photo/Dreamlike.jpg")
        self.image3 = self.image3.resize((140, 140), Image.Resampling.LANCZOS)
        self.image_tk3 = ImageTk.PhotoImage(self.image3)
        self.label = customtkinter.CTkLabel(self.frame3, image=self.image_tk3, text="")
        self.label.pack()

        self.frame4 = customtkinter.CTkFrame(self, width=140, height=140)
        self.frame4.place(relx=0.05, rely=0.45)
        self.image4 = Image.open("photo/StabilityAI.jpg")
        self.image4 = self.image4.resize((140, 140), Image.Resampling.LANCZOS)
        self.image_tk4 = ImageTk.PhotoImage(self.image4)
        self.label = customtkinter.CTkLabel(self.frame4, image=self.image_tk4, text="")
        self.label.pack()

        self.frame5 = customtkinter.CTkFrame(self, width=140, height=140)
        self.frame5.place(relx=0.20, rely=0.45)
        self.image5 = Image.open("photo/Modi.jpg")
        self.image5 = self.image5.resize((140, 140), Image.Resampling.LANCZOS)
        self.image_tk5 = ImageTk.PhotoImage(self.image5)
        self.label = customtkinter.CTkLabel(self.frame5, image=self.image_tk5, text="")
        self.label.pack()

        self.frame6 = customtkinter.CTkFrame(self, width=140, height=140)
        self.frame6.place(relx=0.35, rely=0.45)
        self.image6 = Image.open("photo/PromptHero.jpg")
        self.image6 = self.image6.resize((140, 140), Image.Resampling.LANCZOS)
        self.image_tk6 = ImageTk.PhotoImage(self.image6)
        self.label = customtkinter.CTkLabel(self.frame6, image=self.image_tk6, text="")
        self.label.pack()

        self.frame7 = customtkinter.CTkFrame(self, width=140, height=140)
        self.frame7.place(relx=0.05, rely=0.70)
        self.image7 = Image.open("photo/Arcane.jpg")
        self.image7 = self.image7.resize((140, 140), Image.Resampling.LANCZOS)
        self.image_tk7 = ImageTk.PhotoImage(self.image7)
        self.label = customtkinter.CTkLabel(self.frame7, image=self.image_tk7, text="")
        self.label.pack()

        self.frame8 = customtkinter.CTkFrame(self, width=140, height=140)
        self.frame8.place(relx=0.20, rely=0.70)
        self.image8 = Image.open("photo/Waifu.jpg")
        self.image8 = self.image8.resize((140, 140), Image.Resampling.LANCZOS)
        self.image_tk8 = ImageTk.PhotoImage(self.image8)
        self.label = customtkinter.CTkLabel(self.frame8, image=self.image_tk8, text="")
        self.label.pack()

        self.frame9 = customtkinter.CTkFrame(self, width=140, height=140)
        self.frame9.place(relx=0.35, rely=0.70)
        self.image9 = Image.open("photo/Dreamlike2.jpg")
        self.image9 = self.image9.resize((140, 140), Image.Resampling.LANCZOS)
        self.image_tk9 = ImageTk.PhotoImage(self.image9)
        self.label = customtkinter.CTkLabel(self.frame9, image=self.image_tk9, text="")
        self.label.pack()

        self.radio = tk.StringVar(value="")
        self.radio1 = customtkinter.CTkRadioButton(self, text="CompVis", font=("Helvetica", 20),
                                                   fg_color="purple", variable=self.radio, value="CompVis")
        self.radio1.place(relx=0.05, rely=0.38)
        self.radio1.select()

        self.radio2 = customtkinter.CTkRadioButton(self, text="RunwayML", font=("Helvetica", 20),
                                                   fg_color="purple", variable=self.radio, value="RunwayML")
        self.radio2.place(relx=0.20, rely=0.38)

        self.radio3 = customtkinter.CTkRadioButton(self, text="Dreamlike", font=("Helvetica", 20),
                                                   fg_color="purple", variable=self.radio, value="Dreamlike")
        self.radio3.place(relx=0.35, rely=0.38)

        self.radio4 = customtkinter.CTkRadioButton(self, text="StabilityAI", font=("Helvetica", 20),
                                                   fg_color="purple", variable=self.radio, value="StabilityAI")
        self.radio4.place(relx=0.05, rely=0.63)

        self.radio5 = customtkinter.CTkRadioButton(self, text="Mo-di", font=("Helvetica", 20),
                                                   fg_color="purple", variable=self.radio, value="Mo-di")
        self.radio5.place(relx=0.20, rely=0.63)

        self.radio6 = customtkinter.CTkRadioButton(self, text="PromptHero", font=("Helvetica", 20),
                                                   fg_color="purple", variable=self.radio, value="PromptHero")
        self.radio6.place(relx=0.35, rely=0.63)

        self.radio7 = customtkinter.CTkRadioButton(self, text="Arcane", font=("Helvetica", 20),
                                                   fg_color="purple", variable=self.radio, value="Arcane")
        self.radio7.place(relx=0.05, rely=0.88)

        self.radio8 = customtkinter.CTkRadioButton(self, text="Waifu", font=("Helvetica", 20),
                                                   fg_color="purple", variable=self.radio, value="Waifu")
        self.radio8.place(relx=0.20, rely=0.88)

        self.radio9 = customtkinter.CTkRadioButton(self, text="Dreamlike2", font=("Helvetica", 20),
                                                   fg_color="purple", variable=self.radio, value="Dreamlike2")
        self.radio9.place(relx=0.35, rely=0.88)

        self.goster = customtkinter.CTkLabel(self, height=400, width=400, text=" ")
        self.goster.place(relx=0.79, rely=0.67, anchor=tk.CENTER)

        self.download_button = customtkinter.CTkButton(self, text="Download Image", command=self.download_image)
        self.download_button.place(relx=0.66, rely=0.35, anchor=tk.CENTER)

    def clear_text(self, event):
        if self.textbox.get("1.0", tk.END).strip() == self.default_text:
            self.textbox.delete("1.0", tk.END)

    def restore_text(self, event):
        if not self.textbox.get("1.0", tk.END).strip():
            self.textbox.insert("1.0", self.default_text)

    def switch_event(self):
        if self.switch_var.get() == "on":
            customtkinter.set_appearance_mode("dark")
            self.switch.configure(text="Dark Mode  ")
        if self.switch_var.get() == "off":
            customtkinter.set_appearance_mode("light")
            self.switch.configure(text="Light Mode  ")

    def Generate_function(self):
        self.button.configure(state=tk.DISABLED)

        self.goster.configure(image="")
        self.delete_image()

        self.sayac_lock = threading.Lock()

        image_thread = threading.Thread(target=self.generate_image)
        image_thread.start()

        animation_thread = threading.Thread(target=self.play_animation)
        animation_thread.start()

    def delete_image(self):
        if os.path.exists("generatedimage.png"):
            os.remove("generatedimage.png")

    def set_model_stable(self, model_id):
        self.modelid = model_id
        self.device = "cuda"
        self.pipe = StableDiffusionPipeline.from_pretrained(self.modelid, torch_dtype=torch.float16,
                                                            use_auth_token="TOKEN") #TOKEN kısmına, okunabilir Hugging Face api adresinizi giriniz
        self.pipe.to(self.device)

    def set_model_stable_variant(self, model_id):
        self.modelid = model_id
        self.device = "cuda"
        self.pipe = StableDiffusionPipeline.from_pretrained(self.modelid, variant="fp16", torch_dtype=torch.float16,
                                                            use_auth_token="TOKEN") #TOKEN kısmına, okunabilir Hugging Face api adresinizi giriniz
        self.pipe.to(self.device)

    def generate_image(self):
        selected_radio_value = self.radio.get()

        if selected_radio_value == "CompVis":
            self.set_model_stable_variant("CompVis/stable-diffusion-v1-4")
        elif selected_radio_value == "RunwayML":
            self.set_model_stable_variant("runwayml/stable-diffusion-v1-5")
        elif selected_radio_value == "Dreamlike":
            self.set_model_stable("dreamlike-art/dreamlike-diffusion-1.0")
        elif selected_radio_value == "StabilityAI":
            self.set_model_stable("stabilityai/stable-diffusion-2")
        elif selected_radio_value == "Mo-di":
            self.set_model_stable("nitrosocke/mo-di-diffusion")
        elif selected_radio_value == "PromptHero":
            self.set_model_stable("prompthero/openjourney")
        elif selected_radio_value == "Arcane":
            self.set_model_stable("nitrosocke/Arcane-Diffusion")
        elif selected_radio_value == "Waifu":
            self.set_model_stable("hakurei/waifu-diffusion")
        elif selected_radio_value == "Dreamlike2":
            self.set_model_stable("dreamlike-art/dreamlike-photoreal-2.0")

        prompt = self.textbox.get("1.0", "end-1c")
        translator = Translator(from_lang="tr", to_lang="en")
        prompt_english = translator.translate(prompt)

        with autocast(self.device):
            image = self.pipe(prompt_english).images[0]

        with self.sayac_lock:
            self.sayac = 0
        image.save('generatedimage.png')
        resized_image = image.resize((550, 500), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(resized_image)

        self.after(1000, self.update_image, img)

    def update_image(self, img):
        self.goster.configure(image=img)
        self.button.configure(state=tk.NORMAL)

    def play_animation(self):
        self.loadingtext = tk.Label(self, text="Loading...", font=("Bahnschrift", 15), bg="black", fg="#FFBD09")
        self.loadingtext.place(relx=0.705, rely=0.60, anchor=tk.CENTER)

        self.loading_blocks = []
        for i in range(16):
            block = tk.Label(self, width=2, height=1, bg="#1F2732")
            block.place(x=(i + 46) * 22, y=543)
            self.loading_blocks.append(block)

        self.sayac = 1
        while True:
            if self.sayac == 1:
                for block in self.loading_blocks:
                    block.config(bg="#FFBD09")
                    time.sleep(0.06)
                    self.update_idletasks()
                    block.config(bg="#1F2732")
            else:
                break

        self.loadingtext.destroy()
        for block in self.loading_blocks:
            block.destroy()

    def download_image(self):
        if not os.path.exists("generatedimage.png"):
            tk.messagebox.showwarning("Uyarı", "Lütfen önce bir resim üretin!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            os.rename("generatedimage.png", file_path)

if __name__ == "__main__":
    app = App()
    atexit.register(app.delete_image)
    app.mainloop()