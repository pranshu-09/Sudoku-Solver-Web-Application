# Import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL
from tensorflow.keras.models import load_model
from tabulate import tabulate


# Apply initial preprocessing to sudoku image
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 6)
    threshold_img = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return threshold_img


# Get the main outline of the contour image
def main_outline(contour_img):
    biggest = np.array([])
    max_area = 0

    for i in contour_img:
        area = cv2.contourArea(i)
        if area > 100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)

            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area


# Reframe the sudoku image so that sudoku occupies the whole image removing useless regions
def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4, 1, 2), dtype=np.int32)

    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]

    return points_new


# Split the (9x9) matrix into 81 individual cells
def splitcells(img):
    rows = np.vsplit(img, 9)
    cells = []

    for r in rows:
        cols = np.hsplit(r, 9)
        for cell in cols:
            cells.append(cell)

    return cells


# Crop each cell to keep only the digit in each cell
def CropCell(cells):
    cropped_cells = []

    for image in cells:
        img = np.array(image)
        img = img[6:46, 6:46]
        img = Image.fromarray(img)
        cropped_cells.append(img)

    return cropped_cells

# Backtracking Algorithm to solve sudoku
def is_safe(grid, row, col, num):
    for c in range(9):
        if c!=col and grid[row][c] == num:
            return False

    for r in range(9):
        if r!=row and grid[r][col] == num:
            return False

    sr = row - row % 3
    sc = col - col % 3

    for i in range(3):
        for j in range(3):
            if i+sr!=row and j+sc!=col and grid[i + sr][j + sc] == num:
                return False

    return True

n = 9
def solve_sudoku(grid, row, col):
    if row == n-1 and col==n:
        return True

    if col == n:
        row += 1
        col = 0

    if grid[row][col] > 0:
        return solve_sudoku(grid, row, col + 1)

    for num in range(1, n + 1, 1):
        if is_safe(grid, row, col, num):
            grid[row][col] = num
            if solve_sudoku(grid, row, col + 1):
                return True
        grid[row][col] = 0

    return False

# Function to build sudoku matrix after processing each image
def build_sudoku_matrix(filename):
    sudoku = cv2.imread(f'static\{filename}')
    sudoku = cv2.resize(sudoku, (450, 450))
    thres = preprocess(sudoku)

    contour_1 = sudoku.copy()
    contour_2 = sudoku.copy()

    cont, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_1, cont, -1, (0, 255, 0), 3)

    # Draw main outline of sudoku in the image
    black_img = np.zeros((450, 450, 3), np.uint8)
    biggest, maxArea = main_outline(cont)

    if biggest.size != 0:
        biggest = reframe(biggest)
        cv2.drawContours(contour_2, biggest, -1, (0, 255, 0), 10)

        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imagewrap = cv2.warpPerspective(sudoku, matrix, (450, 450))
        imagewrap = cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    (thresh, bw_img) = cv2.threshold(imagewrap, 150, 255, cv2.THRESH_BINARY)

    # Split cells and crop margins
    sudoku_cells = splitcells(bw_img)
    sudoku_cells_croped = CropCell(sudoku_cells)

    # Load Custom CNN Model for digits prediction
    model = load_model("Model.h5")
    model.load_weights("Model_weights.h5")

    # Build Sudoku Grid
    grid = [[0 for x in range(9)]for y in range(9)]

    # Get predictions and fill digits in sudoku grid
    itr = 0
    for i in range(9):
        for j in range(9):
            img = np.array(sudoku_cells_croped[itr])
            img = img / 255.0

            pred = model.predict(img.reshape(1, 40, 40, 1))

            if np.max(pred) > 0.95:
                grid[i][j] = np.argmax(pred)
            else:
                grid[i][j] = 0

            itr += 1

    return grid

# Function to solve the sudoku grid
def sudoku_result(grid):
    for i in range(0,9):
        for j in range(0,9):
            if grid[i][j]!=0 and is_safe(grid, i, j, grid[i][j])==False:
                return -1

    if solve_sudoku(grid, 0, 0):
        return grid
    else:
        return -1