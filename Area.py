class Area:
    def __init__(self, width, height):
        self.x1 = width
        self.x2 = 0
        self.y1 = height
        self.y2 = 0

    def set(self, x, y):
        if x < self.x1:
            self.x1 = x
        if self.x2 < x:
            self.x2 = x
        if y < self.y1:
            self.y1 = y
        if self.y2 < y:
            self.y2 = y

    def get(self):
        return self.x1, self.y1, self.x2, self.y2

