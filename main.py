import taichi as ti
from random import randint
import GPU
import numpy as np

ti.init(arch=ti.cpu)

window = ti.ui.Window("DLA", res=(500, 500))
canvas = window.get_canvas()

def initDLA(N: int, desiredDensity: float):
    grid = ti.field(dtype=ti.f32, shape=(N, N))
    grid[N//2, N//2] = 1.0
    linkTree = {}
    for _ in range(int(desiredDensity * N**2)):
        createPoint(grid, linkTree)
    print(upscaleGrid(linkTree, N, 2))
    newGrid = ti.field(dtype=ti.f32, shape=(8*2, 8*2))
    Grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]])
    GridB = ti.field(dtype=ti.f32, shape=(8, 8))
    GridB.from_numpy(Grid)
    GPU.upscaleLinear(GridB, 2, newGrid)
    GPU.secondPassBlur(newGrid)
    canvas.set_image(newGrid)
    #print(newGrid)
    return grid

def createPoint(grid, linkTree) :
    N = grid.shape[0]
    x = randint(0, N-1)
    y = randint(0, N-1)
    while not grid[x, y] == 0 :
        x = randint(0, N-1)
        y = randint(0, N-1)

    while grid[x, y] != 1.0 :
        direction = randint(0, 3)
        prevX = x
        prevY = y
        if direction < 1 and x < N - 1:
            x += 1
        elif direction < 2 and x > 0:
            x -= 1
        elif direction < 3 and y < N - 1:
            y += 1
        elif y > 0:
            y -= 1
        if grid[x, y] == 1.0 :
            linkTree[(prevX, prevY)] = [x, y]
            grid[prevX, prevY] = 1.0
            break

def upscaleGrid(linkTree: dict, oldN: int, fact: int) :
    newGrid = ti.field(dtype=ti.i16, shape=(oldN*fact, oldN*fact))
    for key in linkTree.keys() :
        newCube = linkTree[key]
        if key[0] == newCube[0] :
            for i in range(fact+1) :
                newGrid[key[0]*fact, key[1]*fact + i] = 1
        else :
            for i in range(fact+1) :
                newGrid[key[0]*fact + i, key[1]*fact] = 1
    return newGrid

while True :
    initDLA(10, 0.2)
    window.show()

