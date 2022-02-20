import random

from board.car import create_cars
import pygame
from board import tracks

"""

http://rmgi.blog/pygame-2d-car-tutorial.html
https://github.com/tdostilio/Race_Game
https://github.com/ArztSamuel/Applying_EANNs

"""


class Game:
    def __init__(self, num_cars=10, name="No Name", height=768,
                 width=1024, config=None, track='RANDOM'):
        pygame.display.init()
        pygame.display.set_caption(name)
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.num_cars = num_cars
        self.pads = []
        self.config = config if config is None else {}
        self.track = self.choose_random_track() if track == 'RANDOM' else track
        self.set_up_pads()

    def set_up_pads(self):
        if self.track == "HORIZONTAL":
            self.pads = tracks.HORIZONTAL_TRACK
        elif self.track == "SPIRAL":
            self.pads = tracks.SPIRAL_TRACK

    def choose_random_track(self):
        tracks = ["HORIZONTAL", "SPIRAL"]
        return random.choice(tracks)

    def draw_sensors(self, cars):
        for car in cars:
            for sensor in car.sensors_coords:
                pygame.draw.circle(self.screen, (255, 0, 0), sensor, 5)

        pygame.display.update()

    def draw_obstacles(self, car_group):
        # draw board
        self.screen.fill((0, 0, 0))

        pad_group = pygame.sprite.RenderPlain(*self.pads)
        collisions = pygame.sprite.groupcollide(car_group, pad_group, False, False, collided=None)

        for car, obstacle in collisions.items():
            car.crash()

        pad_group.update(collisions)
        pad_group.draw(self.screen)
        car_group.draw(self.screen)

        pygame.display.flip()

    @staticmethod
    def get_car_race_progress(car):
        return (car.position.x/1024)*100

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
                turn = driver.drive(car.sensor_euclidean_distances)
                direction = 1 if turn else -1
                # car.acceleration += acceleration * dt * 10.0
                car.steering = (1 if turn else -1) * dt * 5
                car.steering = max(-car.max_steering, min(car.steering, car.max_steering))
                car.update(dt)
                car.update_history(direction)
                if direction == 1:
                    car.update_stats('positive_turn_count', car.stats.get('positive_turn_count', 0) + 1)
                else:
                    car.update_stats('negative_turn_count', car.stats.get('negative_turn_count', 0) + 1)
                car.update_sensors_coords(self.pads)

            # reset clock timer to count time in between ticks
            self.clock.tick(self.ticks)
            
        pygame.quit()

        return cars