#!/usr/bin/env python3

import json
import math
import os
import time

import numpy as np
import pyopencl as cl
import PIL.Image

STEPS = 1024
XMIN = -2.1
XRANGE = 3.0
YMIN = -1.5
YRANGE = 3.0
DX = XRANGE / STEPS
DY = YRANGE / STEPS

MAX_LOOPS = 10**4
MAX_ITERS_CELLS = 256

SAMPLES = 10**7
MIN_ITERS_SAMPLES = 1*10**6
MAX_ITERS_SAMPLES = 5*10**6
MAX_RENDER_BUFS = 32

PALETTE_LENGTH = 256

def gen_header():
    s = f'''
#pragma once

#define FLOAT double

#define STEPS {STEPS}
#define XMIN {XMIN}
#define XRANGE {XRANGE}
#define YMIN {YMIN}
#define YRANGE {YRANGE}

#define MAX_LOOPS {MAX_LOOPS}
'''
    with open('mandel-generated.h', 'w') as f:
        f.write(s)

def cl_init():
    # setup openCL structs
    ctx = cl.create_some_context(interactive=False)
    print('OpenCL context using devices:')
    for dev in ctx.devices:
        dev_type = cl.device_type.to_string(dev.type)
        print(f' "{dev.name}" on "{dev.platform.name}" type "{dev_type}"')
    cq = cl.CommandQueue(ctx)

    # load kernels
    gen_header()
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    with open('mandel.cl') as f:
        prog = cl.Program(ctx, f.read()).build('-I.')

    return (ctx, cq, prog)

def mandel_iters(ctx, cq, prog, max_iters, x0, y0):
    # intermediate and output arrays
    x = np.array(x0)
    y = np.array(y0)
    iters = np.zeros_like(x0, dtype=np.int32)
    done = np.zeros_like(x0, dtype=np.int32)

    # setup openCL buffers
    mf = cl.mem_flags
    x0_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0)
    y0_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y0)
    x_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
    y_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y)
    iters_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=iters)
    done_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=done)

    # compute iters
    iters_kernel = prog.mandel_iters
    nloops = max(1, max_iters // MAX_LOOPS)
    for n in range(nloops):
        print(f'mandel iters: {n+1} / {nloops}')
        iters_kernel(cq, (math.prod(x0.shape),), None,
                     np.int32(max_iters),
                     x0_d, y0_d, x_d, y_d, iters_d, done_d)
        cq.finish()

    cl.enqueue_copy(cq, iters, iters_d)
    cq.finish()

    return iters

def iters_to_image(ctx, cq, prog, iters, max_iters, palette):
    h, w = iters.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    (palette_len, _channels) = palette.shape

    mf = cl.mem_flags
    iters_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=iters)
    palette_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=palette)
    image_d = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=image)

    prog.iters_to_image(cq, (h*w,), None,
                        np.int32(max_iters), np.int32(palette_len),
                        palette_d, iters_d, image_d)

    cl.enqueue_copy(cq, image, image_d)
    cq.finish()

    return image

def iter_cells(iters, max_iters):
    deltas = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def on_frontier(i, j):
        in_m = [
            iters[i+di, j+dj] == max_iters
            for (di, dj) in deltas
        ]
        return any(in_m) and not all(in_m)

    l = list((i, j)
             for i in range(STEPS-1)
             for j in range(STEPS-1)
             if on_frontier(i, j))
    return l

def sample_cells(ctx, cq, prog, x0, y0, cells):
    per_cell = 1 + SAMPLES // len(cells)
    samples = len(cells) * per_cell
    cell_x = np.array(list(x0[i, j] for (i, j) in cells))
    cell_y = np.array(list(y0[i, j] for (i, j) in cells))
    rand_x = np.random.rand(samples) * DX + np.tile(cell_x, per_cell)
    rand_y = np.random.rand(samples) * DY + np.tile(cell_y, per_cell)

    sample_iters = mandel_iters(ctx, cq, prog,
                                MAX_ITERS_SAMPLES, rand_x, rand_y)
    seeds = list((x, y, i)
                 for (x, y, i) in zip(rand_x, rand_y, map(int, sample_iters))
                 if i > MIN_ITERS_SAMPLES and i < MAX_ITERS_SAMPLES)
    return seeds

def to_unit(n):
    units = [
        (10**9, 'G'),
        (10**6, 'M'),
        (10**3, 'K'),
    ]
    for b, u in units:
        if n >= b:
            return f'{n // b}{u}'
    return f'{n}'

def save_seeds(seeds, fname):
    obj = dict(pointList=[
        {
            'pointX': t[0],
            'pointY': t[1],
            'orbitLength': t[2],
        }
        for t in seeds
    ])
    with open(fname, 'w') as f:
        f.write(json.dumps(obj))

def render_seeds(ctx, cq, prog, seeds):
    # generate N buffers
    # while work to do:
    # - pick buffer, pick unfinished seed
    # - enqueue 1 work item: render seed orbit to buffer up to N loops
    # - when all buffers or seeds "busy", wait
    # - break when all seeds done
    # combine buffers: add each to the first one

    x0 = np.array([t[0] for t in seeds], dtype=np.float64)
    y0 = np.array([t[1] for t in seeds], dtype=np.float64)
    x = np.array(x0)
    y = np.array(y0)
    done = np.zeros((len(seeds),), dtype=np.int32)

    mf = cl.mem_flags
    x0_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0)
    y0_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y0)
    x_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
    y_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y)
    done_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=done)

    nbufs = min(MAX_RENDER_BUFS, len(seeds))
    buffs = []
    buffs_d = []
    for _ in range(nbufs):
        buff = np.zeros((STEPS, STEPS), dtype=np.int32)
        buff_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=buff)
        buffs.append(buff)
        buffs_d.append(buff_d)

    render_kernel = prog.mandel_trace
    while True:
        unfinished = [i for i in range(len(seeds))
                      if done[i] == 0]
        if not(unfinished):
            break
        for seed, buff_d in zip(unfinished, buffs_d):
            render_kernel(cq, (1,), None,
                          np.int32(seed),
                          x0_d, y0_d, x_d, y_d, buff_d, done_d)
        cq.finish()

        cl.enqueue_copy(cq, done, done_d)

    for buff, buff_d in zip(buffs, buffs_d):
        cl.enqueue_copy(cq, buff, buff_d, is_blocking=False)
    cq.finish()

    dest = buffs[0]
    for buff in buffs[1:]:
        dest += buff

    return dest

def counts_to_image(ctx, cq, prog, counts, palette):
    h, w = counts.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    (palette_len, _channels) = palette.shape

    mf = cl.mem_flags
    counts_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=counts)
    palette_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=palette)
    image_d = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=image)

    prog.counts_to_image(cq, (h*w,), None,
                         np.int32(np.max(counts)), np.int32(palette_len),
                         palette_d, counts_d, image_d)

    cl.enqueue_copy(cq, image, image_d)
    cq.finish()

    return image

def main():
    # compute input arrays of point coords
    # NB: xi and yi are arrays
    x0 = np.fromfunction(lambda yi, xi: XMIN + xi * DX,
                         (STEPS, STEPS), dtype=np.float64)
    y0 = np.fromfunction(lambda yi, xi: YMIN + yi * DY,
                         (STEPS, STEPS), dtype=np.float64)

    ctx, cq, prog = cl_init()

    # compute iterations
    iters = mandel_iters(ctx, cq, prog, MAX_ITERS_CELLS, x0, y0)

    # compute image
    palette = np.array([[n, 0, n] for n in range(PALETTE_LENGTH)],
                       dtype=np.uint8)
    image = iters_to_image(ctx, cq, prog,
                           iters, MAX_ITERS_CELLS, palette)

    # write image
    img = PIL.Image.fromarray(image, 'RGB')
    img.save('mandel.png')

    # generate list of cells on border of m-set
    print('generating cell list...')
    cells = iter_cells(iters, MAX_ITERS_CELLS)
    print('cell count:', len(cells))

    # generate samples in cells, retain those with slow escaping orbits
    seeds = sample_cells(ctx, cq, prog, x0, y0, cells)
    print('seed count:', len(seeds))
    suffix = '{}-{}_{}'.format(
        to_unit(SAMPLES),
        to_unit(MIN_ITERS_SAMPLES),
        to_unit(MAX_ITERS_SAMPLES),
    )
    save_seeds(seeds, f'seeds-{suffix}-{int(time.time())}.json')

    # compute per-pixel counts of orbits
    counts = render_seeds(ctx, cq, prog, seeds)

    image2 = counts_to_image(ctx, cq, prog, counts, palette)
    img2 = PIL.Image.fromarray(image2, 'RGB')
    img2.save(f'bbrot-{suffix}.png')

if __name__ == '__main__':
    main()
