import os
import time
from math import tan, radians, degrees, copysign, sqrt

from board.car import Car, create_cars
import pygame
from pygame.math import Vector2

"""

http://rmgi.blog/pygame-2d-car-tutorial.html
https://github.com/tdostilio/Race_Game
https://github.com/ArztSamuel/Applying_EANNs

"""


class PadSprite(pygame.sprite.Sprite):
    normal = pygame.image.load('image/vertical_pads.png')
    def __init__(self, position):
        super(PadSprite, self).__init__()
        self.rect = pygame.Rect(self.normal.get_rect())
        self.rect.center = position
        self.image = self.normal


class HorizontalPad(pygame.sprite.Sprite):
    normal = pygame.image.load('image/race_pads.png')
    def __init__(self, position):
        super(HorizontalPad, self).__init__()
        self.rect = pygame.Rect(self.normal.get_rect())
        self.rect.center = position
        self.image = self.normal


class SmallHorizontalPad(pygame.sprite.Sprite):
    normal = pygame.image.load('image/small_horizontal.png')
    def __init__(self, position):
        super(SmallHorizontalPad, self).__init__()
        self.rect = pygame.Rect(self.normal.get_rect())
        self.rect.center = position
        self.image = self.normal


class SmallVerticalPad(pygame.sprite.Sprite):
    normal = pygame.image.load('image/small_vertical.png')
    def __init__(self, position):
        super(SmallVerticalPad, self).__init__()
        self.rect = pygame.Rect(self.normal.get_rect())
        self.rect.center = position
        self.image = self.normal      


MAZE_PADS = [
    PadSprite((0, 200)),
    PadSprite((0, 400)),
    HorizontalPad((60, 0)),
    HorizontalPad((300, 0)),
    HorizontalPad((700, 0)),
    HorizontalPad((900, 0)),
    PadSprite((1024, 100)),
    PadSprite((1024, 550)),
    HorizontalPad((1024, 768)),
    HorizontalPad((624, 768)),
    HorizontalPad((224, 768)),
    PadSprite((200, 768)),
    PadSprite((200, 368)),
    HorizontalPad((450, 130)),
    HorizontalPad((550, 130)),
    PadSprite((800, 375)),
    SmallHorizontalPad((670, 615)),
    SmallHorizontalPad((470, 615)),
    SmallVerticalPad((350, 490)),
    SmallVerticalPad((350, 390)),
    SmallHorizontalPad((470, 270)),
    SmallVerticalPad((600, 390))
]

STRAIGHT_LINE_PADS = [
    HorizontalPad((1024, 200)),
    HorizontalPad((624, 200)),
    HorizontalPad((224, 200)),
    HorizontalPad((1024, 500)),
    HorizontalPad((624, 500)),
    HorizontalPad((224, 500)),
]

OBSTACLE_PADS = [
    PadSprite((0, 200)),
    PadSprite((0, 400)),
]

class Game:
    def __init__(self, num_cars=10, name="No Name", height=768, width=1024, config={}):
        pygame.init()
        pygame.display.set_caption(name)
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.num_cars = num_cars
        self.pads = []
        self.config=config

    def draw_sensors(self, cars):
        for car in cars:
            for sensor in car.sensors_coords:
                pygame.draw.circle(self.screen, (255, 0, 0), sensor, 5)
                
        pygame.display.update()


    def draw_obstacles(self, car_group):
        # draw board
        self.screen.fill((0, 0, 0))

        # draw obstacles
        self.pads = STRAIGHT_LINE_PADS + OBSTACLE_PADS

        pad_group = pygame.sprite.RenderPlain(*self.pads)
        collisions = pygame.sprite.groupcollide(car_group, pad_group, False, False, collided = None)
        
        for car, obstacle in collisions.items():
            car.crash()

        pad_group.update(collisions)
        pad_group.draw(self.screen)
        car_group.draw(self.screen)

        pygame.display.flip()


    def run_with_neural_networks(self, drivers, time_limit):
        cars = create_cars(self.num_cars, self.config)

        car_group = pygame.sprite.RenderPlain(*cars)

        total_time = 0

        while not self.exit:
            # check if all cars have crashed
            if all(car.crashed for car in cars):
                break

            # check if reached time limit
            if total_time > time_limit:
                break

            # time since last tick
            dt = self.clock.get_time() / 1000
            total_time += dt

            car_group.update(dt)

            self.draw_obstacles(car_group)

            # draw car sensors
            self.draw_sensors(cars)

            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            # update car steering and acceleration
            for car, driver in zip(cars, drivers):
                turn, acceleration = driver.drive(car.sensor_euclidean_distances)
                car.acceleration += acceleration * dt * 10.0
                car.steering += turn * dt * 1.5
                print(car.sensor_euclidean_distances, acceleration, turn)
                car.steering = max(-car.max_steering, min(car.steering, car.max_steering))
                car.update(dt)
                car.update_history()
                car.update_sensors_coords(self.pads)

            # reset clock timer to count time in between ticks
            self.clock.tick(self.ticks)

        pygame.quit()

        return cars


if __name__ == '__main__':
    game = Game()
    game.run()








