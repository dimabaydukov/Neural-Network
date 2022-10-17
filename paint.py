import pygame
from NeuralNetwork import NeuralNetwork
import numpy as np
import cv2
import h5py

data = h5py.File('data/MNISTdata.hdf5', 'r')
train_images = np.float64(data['x_train'][:])
train_labels = np.int64(np.array(data['y_train'][:, 0]))
test_images = np.float64(data['x_test'][:])
test_labels = np.int64(np.array(data['y_test'][:, 0]))
data.close()

num_iterations = 100000
learning_rate = 0.01
image_pixels = 28 * 28
num_outputs = 10
hidden_size = 150

NN = NeuralNetwork(image_pixels, num_outputs, hidden_size)
cost_dict, tests_dict = NN.train(train_images, train_labels, num_iterations, learning_rate)
accuracy = NN.test(test_images, test_labels)

pygame.init()
pygame.display.set_caption("Paint")

window = pygame.display.set_mode((700, 700))
window.fill((255, 255, 255))

brush = pygame.image.load("data/brush.png")
brush = pygame.transform.scale(brush, (20, 20))

number_surface = pygame.Surface((500, 500))
number_surface.fill((0, 0, 0))
number_surface_pos = number_surface.get_rect(center=(350, 350))
window.blit(number_surface, number_surface_pos)

pygame.draw.rect(window, (255, 0, 0), number_surface_pos, 5)

font = pygame.font.SysFont('Arial', 26)

up_surface = font.render("Press LMB to draw, DELETE to clear, ENTER to save, SPACE to predict", True, (0, 0, 0))
up_position = up_surface.get_rect(center=(350, 50))
window.blit(up_surface, up_position)

prediction = ' '

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
                number_surface.fill((0, 0, 0))
                window.blit(number_surface, number_surface_pos)
                pygame.draw.rect(window, (255, 0, 0), number_surface_pos, 5)
                window.blit(up_surface, up_position)
                window.blit(down_surface, down_position)
                pygame.display.update()

            if event.key == pygame.K_RETURN:
                small_image = pygame.transform.smoothscale(number_surface, (28, 28))
                pygame.image.save(small_image, 'data/number.png')

            if event.key == pygame.K_SPACE:
                color_image = cv2.imread('data/number.png')
                gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                n_image = np.divide(gray_image, 255.0)
                array = np.reshape(n_image, (784,))
                prediction = NN.predict_number(array)
                print(prediction)
                down_surface.fill((255, 255, 255))
                window.blit(down_surface, down_position)
                down_surface = font.render("Prediction: " + str(prediction), True, (0, 0, 0))
                window.blit(down_surface, down_position)
                pygame.draw.rect(window, (255, 0, 0), number_surface_pos, 5)
                pygame.display.update()

        if z == 1 and number_surface_pos.collidepoint(x, y):
            number_surface.blit(brush, (x - 105, y - 105))
            window.blit(number_surface, number_surface_pos)
            pygame.draw.rect(window, (255, 0, 0), number_surface_pos, 5)
            pygame.display.update()
