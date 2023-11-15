"""
A simple library for drawing Sudoku grids
"""

from ..sudoku.grid import GridString, Coordinate
from PIL import Image, ImageDraw, ImageFont

# Colors used in the experiment
c_bg_green = (1, 255, 112)
c_bg_blue = (127, 219, 255)
c_bg_purple = (218, 112, 214)
c_bg_red = (255, 69, 0)
c_bg_orange = (255, 165, 0)
c_digit_blue = (0, 0, 255)

line_house_width = 4
line_cell_width = 1


def img_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def img_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def img_add_title(img, title, bg_color=(255, 255, 255)):
    img_title = Image.new('RGB', (img.width, int(.12 * img.height)), color=bg_color)
    draw = ImageDraw.Draw(img_title)
    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', int(.65 * img_title.height))
    text_w, text_h = draw.textsize(title, font=fnt)
    x = int((img_title.width - text_w)) / 2
    y = int((img_title.height - text_h)) / 2
    draw.text((x, y), title, font=fnt, fill='black')
    return img_concat_v(img_title, img)


def get_house_highlights(house_type, goal):
    highlights = {Coordinate(i, goal.y) if house_type.lower() == 'column' else Coordinate(goal.x, i): c_bg_blue for i in
                  range(9)}
    return highlights


def render_sudoku(gridstring, cell_colors={}, digit_colors={}, size=400):
    numbers = GridString.load(gridstring).get_hints()

    # Rendering
    img = Image.new('RGB', (size, size), color=(255, 255, 255))

    img_max = size - 2
    house_width = int(img_max / 3)
    cell_width = int(house_width / 3)

    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', int(.1 * size))
    draw = ImageDraw.Draw(img)

    # Add highlighted cells
    for coord, color in cell_colors.items():
        if color is not None:
            x, y = coord.x, coord.y
            draw.rectangle((y * cell_width,
                            x * cell_width,
                            (y + 1) * cell_width,
                            (x + 1) * cell_width), fill=color)

    # Create exterior borders
    draw.line((0, 0) + (0, img_max), fill=0, width=line_house_width)
    draw.line((0, 0) + (img_max, 0), fill=0, width=line_house_width)
    draw.line((img_max, 0) + (img_max, img_max), fill=0, width=line_house_width)
    draw.line((0, img_max) + (img_max, img_max), fill=0, width=line_house_width)

    # Create house borders
    draw.line((0, house_width) + (img_max, house_width), fill=0, width=line_house_width)
    draw.line((0, 2 * house_width) + (img_max, 2 * house_width), fill=0, width=line_house_width)
    draw.line((house_width, 0) + (house_width, img_max), fill=0, width=line_house_width)
    draw.line((2 * house_width, 0) + (2 * house_width, img_max), fill=0, width=line_house_width)

    # Create cell borders
    for i in range(9):
        draw.line((0, i * cell_width) + (img_max, i * cell_width), fill=0, width=line_cell_width)
        draw.line((i * cell_width, 0) + (i * cell_width, img_max), fill=0, width=line_cell_width)

    # Add numerals
    for coord, num in numbers.items():
        x, y = coord.x, coord.y
        color = digit_colors[coord] if coord in digit_colors else c_digit_blue
        draw.text(((y + .27) * cell_width,
                   (x + .025) * cell_width), str(num),
                  font=fnt, fill=color)

    return img

