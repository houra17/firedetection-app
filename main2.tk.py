import cv2
from playsound import playsound
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from PIL import Image as PILImage
from kivy.clock import Clock
import numpy as np
import time  # Add this line

class FireDetectionApp(App):
    def build(self):
        self.root = BoxLayout(orientation='vertical')

        self.label = Button(text="Fire Detection App", size_hint_y=None, height=40)
        self.root.add_widget(self.label)

        self.start_button = Button(text="Start", on_press=self.start_detection, size_hint_y=None, height=40)
        self.root.add_widget(self.start_button)

        self.stop_button = Button(text="Stop", on_press=self.stop_detection, size_hint_y=None, height=40)
        self.root.add_widget(self.stop_button)
        self.stop_button.disabled = True

        self.video_panel = Image(size_hint=(1, None), height=480)
        self.root.add_widget(self.video_panel)

        self.is_running = False
        self.cooldown_period = 10
        self.last_detection_time = 0
        self.update_delay = 10 
        self.cap = cv2.VideoCapture(0)
        self.fire_cascade = cv2.CascadeClassifier('fire_detection.xml')

        return self.root

    def start_detection(self, instance):
        self.is_running = True
        self.start_button.disabled = True
        self.stop_button.disabled = False
        self.update_video()

    def stop_detection(self, instance):
        self.is_running = False
        self.start_button.disabled = False
        self.stop_button.disabled = True
        self.cap.release()

    def play_sound(self):
        current_time = time.time()
        if current_time - self.last_detection_time > self.cooldown_period:
            print('Fire is detected')
            playsound('audio.mp3')
            self.last_detection_time = current_time

    def update_video(self, *args):
        ret, frame = self.cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fire = self.fire_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in fire:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                self.play_sound()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(frame, (640, 480))

            img = PILImage.fromarray(img)
            buf = np.flip(img, 0).tobytes()  # Use tobytes() instead of tostring()
            texture = Texture.create(size=(img.width, img.height), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.video_panel.texture = texture

        if self.is_running:
            Clock.schedule_once(self.update_video, 1.0 / self.update_delay)
        else:
            self.cap.release()

if __name__ == "__main__":
    FireDetectionApp().run()


