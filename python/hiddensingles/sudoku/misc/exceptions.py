class InvalidEnumException(Exception):
    def __init__(self, arg_name, enum_class, input):
        self.input = input
        self.enum_class = enum_class
        self.message = f"Argument {arg_name} must be part of {enum_class}. Received {input}"


class InvalidWriteException(Exception):

    def __init__(self, x, y, digit, pencilmarks):
        self.x = x
        self.y = y
        self.digit = digit
        self.pencilmarks = pencilmarks
        self.message = f"Cannot write {digit} to ({x}, {y}). Pencilmarks: {pencilmarks}"


class SolutionsVerificationException(Exception):
    def __init__(self, grid, message):
        self.message = f"{message}\n{str(grid)}"
