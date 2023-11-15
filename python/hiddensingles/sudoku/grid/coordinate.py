class Coordinate:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def T(self):
        return Coordinate(self.y, self.x)

    # def toJSON(self):
    #     return {'x': self.x, 'y': self.y}

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.x < other.x or (self.x == other.x and self.y < other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return "({}, {})".format(self.x, self.y)
