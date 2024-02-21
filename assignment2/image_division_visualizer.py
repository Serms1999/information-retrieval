import cv2 as cv


def main():
    color = (0, 0, 255)
    image = cv.imread('./img/two of spades/1.jpg', cv.IMREAD_COLOR)
    x = 40
    cv.line(image, (x, 0), (x, 224), color, 2)
    cv.line(image, (224 - x, 0), (224 - x, 224), color, 2)
    for i in range(3):
        cv.line(image, (x + ((224 - 2*x) // 3) * (i + 1), 0), ((x + ((224 - 2*x) // 3) * (i + 1), 224)), color, 2)

    for i in range(1,  4):
        cv.line(image, (0, 224 // 4 * i), (224, 224 // 4 * i), color, 2)
    cv.imshow('Imagen', image)
    cv.waitKey()


if __name__ == '__main__':
    main()
