import pygame


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


SPIRAL_TRACK = [
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

HORIZONTAL_TRACK = [
    HorizontalPad((1024, 200)),
    HorizontalPad((624, 200)),
    HorizontalPad((224, 200)),
    HorizontalPad((1024, 500)),
    HorizontalPad((624, 500)),
    HorizontalPad((224, 500)),
]
