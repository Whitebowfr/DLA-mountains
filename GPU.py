import taichi as ti

ti.init(arch=ti.vulkan)

@ti.kernel
def upscaleLinear(src: ti.template(), fact: int, newGrid: ti.template()):
    for i, j in src:
        for k in range(fact):
            for l in range(fact):
                x = i + k / fact
                y = j + l / fact
                x0 = int(x)
                y0 = int(y)
                x1 = x0 + 1
                y1 = y0 + 1
                dx = x - x0
                dy = y - y0
                newGrid[i*fact+k, j*fact+l] = (1 - dx) * (1 - dy) * src[x0, y0] + dx * (1 - dy) * src[x1, y0] + (1 - dx) * dy * src[x0, y1] + dx * dy * src[x1, y1]

@ti.kernel
def secondPassBlur(src: ti.template(), newGrid: ti.template()) -> ti.template() :
    for i, j in src:
        newGrid[i, j] = (src[i, j] + src[i+1, j] + src[i, j+1] + src[i+1, j+1]) / 4
    return newGrid
        