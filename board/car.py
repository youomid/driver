from math import tan, radians, degrees
import math
import pygame
from pygame.math import Vector2


class Car(pygame.sprite.Sprite):
    def __init__(self, x, y, image_path, direction=0.0, length=4, max_steering=5.0, max_acceleration=5.0,
                 sensor_distances=None):
        pygame.sprite.Sprite.__init__(self)
        self.starting_position = Vector2(x, y)
        self.position = Vector2(x, y)
        self.velocity = Vector2(50.0, 0.0)
        self.starting_direction = direction
        self.direction = direction
        self.length = length
        # self.max_acceleration = max_acceleration
        self.max_steering = max_steering

        # self.acceleration = 0.0
        self.steering = 0.0
        self.src_image = pygame.image.load(image_path)

        self.crashed = False

        self.sensors_coords = []
        if sensor_distances is None:
            sensor_distances = [(70, 70), (90, 35), (110, 0), (90, -35), (70, -70)]
        # sensor distance from car
        self.sensor_distances = sensor_distances
        self.sensor_euclidean_distances = [self.euclidean_distance(s[0], s[1]) for s in sensor_distances]

        self.create_sensors()

        self.history = []
        self.stats = {}

    def create_sensors(self):
        self.sensors_coords = []
        for sensor_dist in self.sensor_distances:
            self.sensors_coords.append((
                self.position.x + (sensor_dist[0] * math.cos(radians(self.direction)) + sensor_dist[1] * math.sin(
                    radians(self.direction))),
                self.position.y + (sensor_dist[1] * math.cos(radians(self.direction)) - sensor_dist[0] * math.sin(
                    radians(self.direction))),
            ))

    def update_sensors_coords(self, obstacles):
        new_sensors = []
        new_sensor_euclidean_distances = []
        for i, sensor_dist in enumerate(self.sensor_distances):
            new_sensor_coord, sensor_distance = self.calculate_sensor_coords(i, obstacles)
            new_sensors.append(new_sensor_coord)
            new_sensor_euclidean_distances.append(sensor_distance)
        self.sensors_coords = new_sensors
        self.sensor_euclidean_distances = new_sensor_euclidean_distances

    def euclidean_distance(self, x, y):
        return math.sqrt(x ** 2 + y ** 2)

    def calculate_sensor_coords(self, sensor_id, obstacles):
        new_sensor_coord = (
            self.position.x + (self.sensor_distances[sensor_id][0] * math.cos(radians(self.direction)) +
                               self.sensor_distances[sensor_id][1] * math.sin(radians(self.direction))),
            self.position.y + (self.sensor_distances[sensor_id][1] * math.cos(radians(self.direction)) -
                               self.sensor_distances[sensor_id][0] * math.sin(radians(self.direction))),
        )

        collision_point, collision_distance = self.calculate_collision_point(new_sensor_coord, obstacles)
        return (collision_point.x, collision_point.y), collision_distance

    def check_sensor_collision(self, new_sensor_coord, obstacles):
        for obstacle in obstacles:
            if obstacle.rect.collidepoint(Vector2(new_sensor_coord[0], new_sensor_coord[1])):
                return obstacle

        return None

    def calculate_collision_point(self, sensor_pos, obstacles):
        current_pos = Vector2(self.position)
        heading = Vector2(sensor_pos[0], sensor_pos[1]) - self.position
        direction = heading.normalize()

        for i in range(int(heading.length())):
            current_pos += direction
            for obstacle in obstacles:
                if obstacle.rect.collidepoint(current_pos):
                    return current_pos, current_pos.distance_to(Vector2(self.position))

        return Vector2(sensor_pos[0], sensor_pos[1]), current_pos.distance_to(Vector2(self.position))

    def crash(self):
        self.stop()
        self.crashed = True

    def stop(self):
        # self.acceleration = 0
        self.steering = 0.0
        self.velocity = Vector2(0.0, 0.0)

    def reset(self):
        self.position = self.starting_position
        self.direction = self.starting_direction
        self.stop()

    def update(self, dt):

        if not self.crashed:
            # self.velocity += (self.acceleration * dt, 0)
            if self.steering:
                turning_radius = self.length / tan(radians(self.steering))
                angular_velocity = self.velocity.x / turning_radius
            else:
                angular_velocity = 0

            self.position += self.velocity.rotate(-self.direction) * dt
            self.direction += degrees(angular_velocity) * dt

            self.image = pygame.transform.rotate(self.src_image, self.direction)
            self.rect = self.image.get_rect()
            self.rect.center = self.position

        return self.position

    def update_history(self, direction):
        self.history.append({
            'sensor0': self.sensor_euclidean_distances[0],
            'sensor1': self.sensor_euclidean_distances[1],
            'sensor2': self.sensor_euclidean_distances[2],
            'sensor3': self.sensor_euclidean_distances[3],
            'sensor4': self.sensor_euclidean_distances[4],
            # 'acceleration': self.acceleration,
            'steering': self.steering,
            'direction': direction
        })

    def update_stats(self, key, value):
        self.stats[key] = value


def create_cars(num_cars, config):
    cars = []
    for i in range(num_cars):
        cars.append(Car(90, 350, "image/car.png", direction=0.0, sensor_distances=config.get('sensor_distances')))

    return cars
