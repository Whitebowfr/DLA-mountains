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
def secondPassBlur(src: ti.template()) :
    for i, j in src:
        sum = 0.0
        if i > 0 :
            sum += src[i-1, j]
        if i < src.shape[0] - 1 :
            sum += src[i+1, j]
        if j > 0 :
            sum += src[i, j-1]
        if j < src.shape[1] - 1 :
            sum += src[i, j+1]
        src[i, j] = sum / 4
        