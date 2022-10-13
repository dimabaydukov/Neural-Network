import pygame
import pickle
from NeuralNetwork import NeuralNetwork
import h5py
import numpy as np

data = h5py.File('data/MNISTdata.hdf5', 'r')
test_labels = np.int64(np.array(data['y_test'][:, 0]))
data.close()

NN = NeuralNetwork(28 * 28, 10, 300)
with open('data/save.txt', 'rb+') as t:
    for j in range(3):
        if j == 0:
            NN = pickle.load(t)

pygame.init()
pygame.display.set_caption("Paint")

window = pygame.display.set_mode((700, 700))
window.fill((255, 255, 255))

brush = pygame.image.load("data/brush.png")
brush = pygame.transform.scale(brush, (10, 10))

number_surface = pygame.Surface((500, 500))
number_surface.fill((255, 255, 255))
number_surface_pos = number_surface.get_rect(center=(350, 350))
window.blit(number_surface, number_surface_pos)

pygame.draw.rect(window, (0, 0, 0), number_surface_pos, 5)

font = pygame.font.SysFont('Arial', 26)

up_surface = font.render("Press LMB to draw, DELETE to clear, ENTER to save, SPACE to predict", True, (0, 0, 0))
up_position = up_surface.get_rect(center=(350, 50))
window.blit(up_surface, up_position)

prediction = '5'

down_surface = font.render("Prediction: " + prediction, True, (0, 0, 0))
down_position = down_surface.get_rect(center=(350, 650))
window.blit(down_surface, down_position)

pygame.display.update()
clock = pygame.time.Clock()

z = 0

while 1:
    clock.tick(60)
    x, y = pygame.mouse.get_pos()

    pygame.draw.rect(window, (0, 0, 0), number_surface_pos, 5)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN and number_surface_pos.collidepoint(x, y):
            if event.button == 1:
                z = 1
        elif event.type == pygame.MOUSEBUTTONUP:
            z = 0
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_DELETE:
                number_surface.fill((255, 255, 255))
                window.blit(number_surface, number_surface_pos)
                pygame.draw.rect(window, (0, 0, 0), number_surface_pos, 5)
                window.blit(up_surface, up_position)
                window.blit(down_surface, down_position)
                pygame.display.update()

            if event.key == pygame.K_RETURN:
                pygame.image.save(number_surface, 'data/number.png')

            if event.key == pygame.K_SPACE:
                pass

        if z == 1 and number_surface_pos.collidepoint(x, y):
            number_surface.blit(brush, (x - 105, y - 105))
            window.blit(number_surface, number_surface_pos)
            pygame.draw.rect(window, (0, 0, 0), number_surface_pos, 5)
            window.blit(up_surface, up_position)
            window.blit(down_surface, down_position)
            pygame.display.update()
