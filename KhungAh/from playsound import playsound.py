import pygame

# Khởi tạo pygame
pygame.mixer.init()
pygame.mixer.music.load("sound/alarm.mp3")
pygame.mixer.music.play()

# Đợi cho đến khi âm thanh kết thúc
while pygame.mixer.music.get_busy():
    continue
