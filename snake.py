from typing import List, Set
from dataclasses import dataclass

import numpy as np
import pygame
from enum import Enum, unique
import sys
import random


FPS = 15

INIT_LENGTH = 4

WIDTH = 480
HEIGHT = 480
GRID_SIDE = 24
GRID_WIDTH = WIDTH // GRID_SIDE
GRID_HEIGHT = HEIGHT // GRID_SIDE

BRIGHT_BG = (103, 223, 235)
DARK_BG = (78, 165, 173)

SNAKE_COL = (6, 38, 7)
FOOD_COL = (224, 160, 38)
OBSTACLE_COL = (209, 59, 59)
VISITED_COL = (24, 42, 142)


@unique
class Direction(tuple, Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    def reverse(self):
        x, y = self.value
        return Direction((x * -1, y * -1))


@dataclass
class Position:
    x: int
    y: int

    def check_bounds(self, width: int, height: int):
        return (self.x >= width) or (self.x < 0) or (self.y >= height) or (self.y < 0)

    def draw_node(self, surface: pygame.Surface, color: tuple, background: tuple):
        r = pygame.Rect(
            (int(self.x * GRID_SIDE), int(self.y * GRID_SIDE)), (GRID_SIDE, GRID_SIDE)
        )
        pygame.draw.rect(surface, color, r)
        pygame.draw.rect(surface, background, r, 1)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Position):
            return (self.x == o.x) and (self.y == o.y)
        else:
            return False

    def __str__(self):
        return f"X{self.x};Y{self.y};"

    def __hash__(self):
        return hash(str(self))


class GameNode:
    nodes: Set[Position] = set()

    def __init__(self):
        self.position = Position(0, 0)
        self.color = (0, 0, 0)

    def randomize_position(self):
        try:
            GameNode.nodes.remove(self.position)
        except KeyError:
            pass

        condidate_position = Position(
            random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1),
        )

        if condidate_position not in GameNode.nodes:
            self.position = condidate_position
            GameNode.nodes.add(self.position)
        else:
            self.randomize_position()

    def draw(self, surface: pygame.Surface):
        self.position.draw_node(surface, self.color, BRIGHT_BG)


class Food(GameNode):
    def __init__(self):
        super(Food, self).__init__()
        self.color = FOOD_COL
        self.randomize_position()


class Obstacle(GameNode):
    def __init__(self):
        super(Obstacle, self).__init__()
        self.color = OBSTACLE_COL
        self.randomize_position()


class Snake:
    def __init__(self, screen_width, screen_height, init_length):
        self.color = SNAKE_COL
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.init_length = init_length
        self.reset()

    def reset(self):
        self.length = self.init_length
        self.positions = [Position((GRID_SIDE // 2), (GRID_SIDE // 2))]
        self.direction = random.choice([e for e in Direction])
        self.score = 0
        self.hasReset = True

    def get_head_position(self) -> Position:
        return self.positions[0]

    def turn(self, direction: Direction):
        if self.length > 1 and direction.reverse() == self.direction:
            return
        else:
            self.direction = direction

    def move(self):
        self.hasReset = False
        cur = self.get_head_position()
        x, y = self.direction.value
        new = Position(cur.x + x, cur.y + y,)
        if self.collide(new):
            self.reset()
        else:
            self.positions.insert(0, new)
            while len(self.positions) > self.length:
                self.positions.pop()

    def collide(self, new: Position):
        return (new in self.positions) or (new.check_bounds(GRID_WIDTH, GRID_HEIGHT))

    def eat(self, food: Food):
        if self.get_head_position() == food.position:
            self.length += 1
            self.score += 1
            while food.position in self.positions:
                food.randomize_position()

    def hit_obstacle(self, obstacle: Obstacle):
        if self.get_head_position() == obstacle.position:
            self.length -= 1
            self.score -= 1
            if self.length == 0:
                self.reset()

    def draw(self, surface: pygame.Surface):
        for p in self.positions:
            p.draw_node(surface, self.color, BRIGHT_BG)


class Player:
    def __init__(self) -> None:
        self.visited_color = VISITED_COL
        self.visited: Set[Position] = set()
        self.chosen_path: List[Direction] = []

    def move(self, snake: Snake) -> bool:
        try:
            next_step = self.chosen_path.pop(0)
            snake.turn(next_step)
            return False
        except IndexError:
            return True

    def search_path(self, snake: Snake, food: Food, *obstacles: Set[Obstacle]):
        """
        Do nothing, control is defined in derived classes
        """
        pass

    def turn(self, direction: Direction):
        """
        Do nothing, control is defined in derived classes
        """
        pass

    def draw_visited(self, surface: pygame.Surface):
        for p in self.visited:
            p.draw_node(surface, self.visited_color, BRIGHT_BG)


class SnakeGame:
    def __init__(self, snake: Snake, player: Player) -> None:
        pygame.init()
        pygame.display.set_caption("AIFundamentals - SnakeGame")

        self.snake = snake
        self.food = Food()
        self.obstacles: Set[Obstacle] = set()
        for _ in range(40):
            ob = Obstacle()
            while any([ob.position == o.position for o in self.obstacles]):
                ob.randomize_position()
            self.obstacles.add(ob)

        self.player = player

        self.fps_clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode(
            (snake.screen_height, snake.screen_width), 0, 32
        )
        self.surface = pygame.Surface(self.screen.get_size()).convert()
        self.myfont = pygame.font.SysFont("monospace", 16)

    def drawGrid(self):
        for y in range(0, int(GRID_HEIGHT)):
            for x in range(0, int(GRID_WIDTH)):
                p = Position(x, y)
                if (x + y) % 2 == 0:
                    p.draw_node(self.surface, BRIGHT_BG, BRIGHT_BG)
                else:
                    p.draw_node(self.surface, DARK_BG, DARK_BG)

    def run(self):
        while not self.handle_events():
            self.fps_clock.tick(FPS)
            self.drawGrid()
            if self.player.move(self.snake) or self.snake.hasReset:
                self.player.search_path(self.snake, self.food, self.obstacles)
                self.player.move(self.snake)
            self.snake.move()
            self.snake.eat(self.food)
            for ob in self.obstacles:
                self.snake.hit_obstacle(ob)
            for ob in self.obstacles:
                ob.draw(self.surface)
            self.player.draw_visited(self.surface)
            self.snake.draw(self.surface)
            self.food.draw(self.surface)
            self.screen.blit(self.surface, (0, 0))
            text = self.myfont.render(
                "Score {0}".format(self.snake.score), 1, (0, 0, 0)
            )
            self.screen.blit(text, (5, 10))
            pygame.display.update()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_UP:
                    self.player.turn(Direction.UP)
                elif event.key == pygame.K_DOWN:
                    self.player.turn(Direction.DOWN)
                elif event.key == pygame.K_LEFT:
                    self.player.turn(Direction.LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.player.turn(Direction.RIGHT)
        return False


class HumanPlayer(Player):
    def __init__(self):
        super(HumanPlayer, self).__init__()

    def turn(self, direction: Direction):
        self.chosen_path.append(direction)


# ----------------------------------
# DO NOT MODIFY CODE ABOVE THIS LINE
# ----------------------------------
from collections import deque
from queue import Queue
from time import sleep


class SearchBasedPlayer(Player):
    def __init__(self, algorithm):
        self.algorithm = algorithm
        super(SearchBasedPlayer, self).__init__()

    def move(self, snake: Snake) -> bool:
        try:
            next_step = self.chosen_path.pop(0)
            snake.turn(next_step)
            return False
        except IndexError:
            return True

    def search_path(self, snake: Snake, food: Food, *obstacles: Set[Obstacle]):
        self.visited.clear()
        obstacles_positions = set([obstacle.position for obstacle in obstacles[0]])
        bfs = self.algorithm(20, 20, obstacles_positions)
        path, visited = bfs.search(
            head_position=snake.get_head_position(),
            snake_positions=snake.positions,
            snake_direction=snake.direction,
            obstacles_positions=obstacles_positions,
            food_position=food.position)

        directions = []
        print("Path", path)
        for i in range(len(path) - 1):
            direction_val = (path[i + 1].x - path[i].x, path[i + 1].y - path[i].y)

            direction = None
            if direction_val == Direction.RIGHT.value:
                direction = Direction.RIGHT
            elif direction_val == Direction.LEFT.value:
                direction = Direction.LEFT
            elif direction_val == Direction.DOWN.value:
                direction = Direction.DOWN
            elif direction_val == Direction.UP.value:
                direction = Direction.UP
            directions.append(direction)

        self.chosen_path: List[Direction] = directions
        self.visited = visited - obstacles[0]


class SearchAlgorithm:
    def __init__(self, grid_size_x: int, grid_size_y: int, obstacles_positions: list[Position]):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.obstacles_positions = obstacles_positions

    def search(self,
               head_position: Position,
               snake_positions: list[Position],
               snake_direction: Direction,
               obstacles_positions: list[Position],
               food_position: Position):
        pass

    def expand(self, position, food_position: Position, snake_direction: Direction, snake_positions, visited, obstacles_positions) -> tuple[bool, list]:
        old_x = position.x
        old_y = position.y
        new_positions = []
        for direction in Direction:

            new_position = Position(old_x + direction[0], old_y + direction[1])
            if self.is_solution(new_position, food_position):
                return True, [new_position]
            if self.is_possible(new_position, snake_positions, obstacles_positions) and self.is_novel(new_position, visited):
                new_positions.append(new_position)

        return False, new_positions

    def is_possible(self, position, snake_positions, obstacles_positions) -> bool:
        """

        :param obstacles_positions:
        :param position:
        :param snake_positions:
        :return: True for possible move, else False
        """
        return (
            position not in snake_positions
            and not position.check_bounds(self.grid_size_x, self.grid_size_y)
            and position not in obstacles_positions
        )

    @staticmethod
    def is_novel(position, visited) -> bool:
        return position not in visited

    @staticmethod
    def is_solution(position, food_position) -> bool:
        return food_position == position


class BFS(SearchAlgorithm):
    def __init__(self, grid_size_x: int, grid_size_y: int, obstacles_positions: list[Position]):
        super().__init__(grid_size_x, grid_size_y, obstacles_positions)

    def search(self,
               head_position: Position,
               snake_positions: list[Position],
               snake_direction: Direction,
               obstacles_positions: list[Position],
               food_position: Position):

        visited = set()
        candidates = Queue()
        candidates.put([head_position])

        while True:
            candidate = candidates.get()
            candidate_last = candidate[-1]
            is_food, positions = self.expand(candidate_last, food_position, snake_direction, snake_positions, visited,
                                             obstacles_positions)
            if is_food:
                candidate.append(*positions)
                return candidate, visited
            for position in positions:
                visited.add(position)
                new_candidate = candidate.copy()
                new_candidate.append(position)
                candidates.put(new_candidate)
            if candidates.empty():
                return [], visited


class DFS(SearchAlgorithm):
    def __init__(self, grid_size_x: int, grid_size_y: int, obstacles_positions: list[Position]):
        super().__init__(grid_size_x, grid_size_y, obstacles_positions)

    def search(self,
               head_position: Position,
               snake_positions: list[Position],
               snake_direction: Direction,
               obstacles_positions: list[Position],
               food_position: Position):

        visited = set()
        candidates = list()
        candidates.append([head_position])

        while True:
            if candidates is None:
                return [], visited
            candidate = candidates.pop()
            candidate_last = candidate[-1]
            is_food, positions = self.expand(candidate_last, food_position, snake_direction, snake_positions, visited,
                                             obstacles_positions)
            if is_food:
                candidate.append(*positions)
                return candidate, visited
            for position in positions:
                visited.add(position)
                new_candidate = candidate.copy()
                new_candidate.append(position)
                candidates.append(new_candidate)


class DirectedPosition:
    def __init__(self, position, parent, score=np.inf):
        self.directed_position = (position, score, parent)

    def __hash__(self):
        return self.directed_position[0].__hash__()

    def __getitem__(self, item: int):
        return self.directed_position[item]


from itertools import product
import math


class PriorityQueue:
    def __init__(self, array=None, max_size=100):
        if array is None:
            self.heap_size = 0
            self.heap_array = [None for _ in range(max_size)]
        else:
            self.heap_array = array
            self.heap_size = len(array)
            self.__build_heap()

    def insert(self, item):
        node_index = self.heap_size
        self.heap_size += 1

        while node_index > 0 and self.heap_array[self.__parent(node_index)][1] > item[1]:
            self.heap_array[node_index] = self.heap_array[self.__parent(node_index)]
            node_index = self.__parent(node_index)
        self.heap_array[node_index] = item

    def extract_min(self):
        if not self.heap_size:
            return None
        minimum = self.heap_array[0]
        self.heap_size -= 1
        self.heap_array[0], self.heap_array[self.heap_size] = self.heap_array[self.heap_size], self.heap_array[0]
        self.__heapify(0)
        self.heap_array[self.heap_size] = None
        return minimum

    def __build_heap(self):
        for index in range(self.heap_size // 2 - 1, -1, -1):
            self.__heapify(index)

    def __heapify(self, node_index):
        left_child_index = self.__left_child(node_index)
        right_child_index = self.__right_child(node_index)
        if left_child_index < self.heap_size and self.heap_array[left_child_index][1] < self.heap_array[node_index][1]:
            smallest_index = left_child_index
        else:
            smallest_index = node_index

        if (right_child_index < self.heap_size
                and self.heap_array[right_child_index][1] < self.heap_array[smallest_index][1]):
            smallest_index = right_child_index

        if smallest_index != node_index:
            self.heap_array[node_index], self.heap_array[smallest_index] = (self.heap_array[smallest_index],
                                                                            self.heap_array[node_index])
            self.__heapify(smallest_index)

    @staticmethod
    def __parent(node_index):
        return math.ceil((node_index - 1) / 2)

    @staticmethod
    def __left_child(node_index):
        return 2 * node_index + 1

    @staticmethod
    def __right_child(node_index):
        return 2 * (node_index + 1)


class Dijkstra(SearchAlgorithm):
    def __init__(self, grid_size_x: int, grid_size_y: int, obstacles_positions: list[Position]):
        super().__init__(grid_size_x, grid_size_y, obstacles_positions)

    def expand(self, position, food_position: Position, snake_direction: Direction, snake_positions, visited, obstacles_positions) -> tuple[bool, list]:
        old_x = position.x
        old_y = position.y
        new_positions = []
        for direction in Direction:

            new_position = Position(old_x + direction[0], old_y + direction[1])
            if self.is_solution(new_position, food_position):
                return True, [new_position]
            if self.is_possible(new_position, snake_positions, obstacles_positions) and self.is_novel(new_position, visited):
                new_positions.append(new_position)

        return False, new_positions

    def is_possible(self, position, snake_positions, obstacles_positions) -> bool:
        """

        :param position:
        :param snake_positions:
        :return: True for possible move, else False
        """
        return (
            position not in snake_positions
            and not position.check_bounds(self.grid_size_x, self.grid_size_y)
        )

    def search(self,
               head_position: Position,
               snake_positions: list[Position],
               snake_direction: Direction,
               obstacles_positions: list[Position],
               food_position: Position) -> list[Position]:

        visited = set()
        candidates = PriorityQueue(max_size=100000)
        candidates.insert(DirectedPosition(head_position, None, 0))

        while candidates.heap_size != 0:
            candidate = candidates.extract_min()
            if candidate[0] in visited:
                continue
            visited.add(candidate[0])

            is_food, positions = self.expand(candidate[0], food_position, snake_direction, snake_positions, visited,
                                             obstacles_positions)

            if is_food:
                position = candidate
                path = [*positions]
                while True:
                    path.append(position[0])
                    if position[2] is None:
                        path.reverse()
                        return path, visited
                    position = position[2]

            #print(len(positions))
            for position in positions:
                new_score = candidate[1] + 1
                if position in obstacles_positions:
                    new_score += 20
                new_candidate = DirectedPosition(position, candidate, new_score)
                candidates.insert(new_candidate)
        return [], visited


class AStar(SearchAlgorithm):
    def __init__(self, grid_size_x: int, grid_size_y: int, obstacles_positions: list[Position]):
        super().__init__(grid_size_x, grid_size_y, obstacles_positions)

    def expand(self, position, food_position: Position, snake_direction: Direction, snake_positions, visited, obstacles_positions) -> tuple[bool, list]:
        old_x = position.x
        old_y = position.y
        new_positions = []
        for direction in Direction:

            new_position = Position(old_x + direction[0], old_y + direction[1])
            if self.is_solution(new_position, food_position):
                return True, [new_position]
            if self.is_possible(new_position, snake_positions, obstacles_positions) and self.is_novel(new_position, visited):
                new_positions.append(new_position)

        return False, new_positions

    def is_possible(self, position, snake_positions, obstacles_positions) -> bool:
        return (
            position not in snake_positions
            and not position.check_bounds(self.grid_size_x, self.grid_size_y)
        )

    def search(self,
               head_position: Position,
               snake_positions: list[Position],
               snake_direction: Direction,
               obstacles_positions: list[Position],
               food_position: Position):

        visited = set()
        candidates = PriorityQueue(max_size=10000000)
        candidates.insert(DirectedPosition(head_position, None, 0))

        while candidates.heap_size != 0:
            candidate = candidates.extract_min()
            if candidate[0] in visited:
                continue
            visited.add(candidate[0])

            is_food, positions = self.expand(candidate[0], food_position, snake_direction, snake_positions, visited,
                                             obstacles_positions)

            if is_food:
                position = candidate
                path = [*positions]
                while True:
                    path.append(position[0])
                    if position[2] is None:
                        path.reverse()
                        return path, visited
                    position = position[2]

            for position in positions:
                new_score = (candidate[1]
                             - np.sqrt((food_position.x - candidate[0].x) ** 2 + (food_position.y - candidate[0].y) ** 2)
                             + 1
                             + np.sqrt((food_position.x - position.x) ** 2 + (food_position.y - position.y) ** 2))

                if position in obstacles_positions:
                    new_score += 20
                new_candidate = DirectedPosition(position, candidate, new_score)
                candidates.insert(new_candidate)
        return [], visited


if __name__ == "__main__":
    snake_ = Snake(WIDTH, WIDTH, INIT_LENGTH)
    #  player_ = HumanPlayer()
    player_ = SearchBasedPlayer(AStar)
    game = SnakeGame(snake_, player_)
    game.run()
