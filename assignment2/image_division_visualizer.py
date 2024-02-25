import random

import cv2 as cv


def main():
    image = cv.imread('./img/seven of spades/1.jpg', cv.IMREAD_COLOR)
    overlay = image.copy()


    widths = [40, 48, 48, 48, 40]
    heights = [56, 56, 56, 56]
    alpha = 0.5

    # Create a dictionary with the colors
    # (0, 0) and (4, 3) have color yellow
    # (1, 1) to (3, 1) have color red and also (1, 2) to (3, 2)
    # the rest have color green
    """
    colors = {
        (0, 0): (0, 255, 255),
        (4, 3): (0, 255, 255),
        (1, 1): (0, 0, 255),
        (2, 1): (0, 0, 255),
        (3, 1): (0, 0, 255),
        (1, 2): (0, 0, 255),
        (2, 2): (0, 0, 255),
        (3, 2): (0, 0, 255),
    }

    for i, w in enumerate(widths):
        for j, h in enumerate(heights):
            x = sum(widths[:i])
            y = sum(heights[:j])
            # Pick a random color
            color = colors.get((i, j), (0, 255, 0))
            cv.rectangle(overlay, (x, y), (x + w, y + h), color, cv.FILLED)

    image3 = cv.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    image3 = image.copy()

    x = 40
    color = (0, 0, 0)
    cv.line(image3, (x, 0), (x, 224), color, 2)
    cv.line(image3, (224 - x, 0), (224 - x, 224), color, 2)
    for i in range(3):
        cv.line(image3, (x + ((224 - 2 * x) // 3) * (i + 1), 0), ((x + ((224 - 2 * x) // 3) * (i + 1), 224)), color, 2)

    for i in range(1, 4):
        cv.line(image3, (0, 224 // 4 * i), (224, 224 // 4 * i), color, 2)

    cv.imshow('Imagen', image3)
    cv.waitKey()

    cv.imwrite('output.jpg', image3)
    """
    cv.imwrite('output.jpg', image[0:56, 0:40, :])


if __name__ == '__main__':
    main()
