import pygame
import pickle
from NeuralNetwork import NeuralNetwork


NN = NeuralNetwork
with open('data/save.txt', 'rb+') as t:
    for j in range(3):
        if j == 0:
            NN = pickle.load(t)

pygame.init()

window = pygame.display.set_mode((700, 700))

window.fill((255, 255, 255))

brush = pygame.image.load("data/brush.png")
brush = pygame.transform.scale(brush, (10, 10))

pygame.display.update()
clock = pygame.time.Clock()

z = 0

while 1:
    clock.tick(60)
    x, y = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            z = 1
        elif event.type == pygame.MOUSEBUTTONUP:
            z = 0

        if z == 1:
            window.blit(brush, (x-5, y-5))
            pygame.display.update()
