#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys
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
MAX_RENDER_BUF_MEM = 2*1024**3
MAX_RENDER_BUFS = MAX_RENDER_BUF_MEM // (4 * STEPS * STEPS)

ANIMATE_FPS = 25
ANIMATE_SECONDS = 10

CL_HEADER_FILE_NAME = 'bbrot-generated.h'

def gen_header(fname):
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
    with open(fname, 'w') as f:
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
    gen_header(CL_HEADER_FILE_NAME)
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    with open('bbrot.cl') as f:
        prog = cl.Program(ctx, f.read()).build('-I.')

    os.unlink(CL_HEADER_FILE_NAME)
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
    nloops = math.ceil(max_iters / MAX_LOOPS)
    for n in range(nloops):
        print(f'mandel iters: {n+1} / {nloops}')
        iters_kernel(cq, (math.prod(x0.shape),), None,
                     np.int32(max_iters),
                     x0_d, y0_d, x_d, y_d, iters_d, done_d)
        cq.finish()

    cl.enqueue_copy(cq, iters, iters_d)
    cq.finish()

    return iters

def frontier_cells(iters, max_iters):
    # Return coordinates for cells that have some corners inside the
    # M-set and some corners outside.
    # This uses numpy arrays operations for speed, as the more
    # readable code in regular Python is way slower.

    iters_maxed = np.int8(iters == max_iters)
    cell_corners_maxed = \
        iters_maxed[ :-1,  :-1] + \
        iters_maxed[1:,    :-1] + \
        iters_maxed[ :-1, 1:  ] + \
        iters_maxed[1:,   1:  ]
    cell_on_border = (cell_corners_maxed > 0) * (cell_corners_maxed < 4)
    selected = list(zip(*np.nonzero(cell_on_border)))

    return selected

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

def load_seeds(fname):
    with open(fname) as f:
        obj = json.load(f)
    l = [ (o['pointX'], o['pointY'], o['orbitLength'])
          for o in obj['pointList'] ]
    return l

def render_seeds_gen(ctx, cq, prog, seeds, iter_checkpoints):
    # generate N buffers
    # while work to do:
    # - pick buffer, pick unfinished seed
    # - enqueue k<=N work items: render seed orbit to buffer up to L loops
    # - when all buffers or seeds "busy", wait
    # - break when all seeds done
    # combine buffers: add each to the first one

    nbufs = min(MAX_RENDER_BUFS, len(seeds))

    seed_list = np.zeros((nbufs,), dtype=np.int32)
    x0 = np.array([t[0] for t in seeds], dtype=np.float64)
    y0 = np.array([t[1] for t in seeds], dtype=np.float64)
    x = np.array(x0)
    y = np.array(y0)
    buff = np.zeros((nbufs, STEPS, STEPS), dtype=np.int32)
    iters = np.zeros_like(x0, dtype=np.int32)
    done = np.zeros_like(x0, dtype=np.int32)

    mf = cl.mem_flags
    seed_list_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=seed_list)
    x0_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0)
    y0_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y0)
    x_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
    y_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y)
    buff_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=buff)
    iters_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=iters)
    done_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=done)

    mandel_trace = prog.mandel_trace
    for max_iters in iter_checkpoints:
        while True:
            unfinished = [i for i in range(len(seeds))
                          if done[i] == 0]
            if not(unfinished):
                break

            n = min(nbufs, len(unfinished))
            for buff_id in range(n):
                seed_list[buff_id] = unfinished[buff_id]
            cl.enqueue_copy(cq, seed_list_d, seed_list)

            mandel_trace(cq, (n,), None,
                         np.int32(max_iters),
                         seed_list_d, x0_d, y0_d, x_d, y_d,
                         buff_d, iters_d, done_d)
            cq.finish()

            cl.enqueue_copy(cq, done, done_d)

        cl.enqueue_copy(cq, buff, buff_d)

        counts = np.sum(buff, axis=0, dtype=np.int32)
        yield counts

        done *= 0
        cl.enqueue_copy(cq, done_d, done)

def render_seeds(ctx, cq, prog, seeds):
    for counts in render_seeds_gen(ctx, cq, prog, seeds, [-1]):
        return counts

def compute():
    # compute input arrays of point coords
    # NB: xi and yi are arrays
    x0 = np.fromfunction(lambda yi, xi: XMIN + xi * DX,
                         (STEPS, STEPS), dtype=np.float64)
    y0 = np.fromfunction(lambda yi, xi: YMIN + yi * DY,
                         (STEPS, STEPS), dtype=np.float64)

    ctx, cq, prog = cl_init()

    # compute iterations
    iters = mandel_iters(ctx, cq, prog, MAX_ITERS_CELLS, x0, y0)

    # generate list of cells on border of m-set
    print('generating cell list...')
    cells = frontier_cells(iters, MAX_ITERS_CELLS)
    print('cell count:', len(cells))

    # generate samples in cells, retain those with slow escaping orbits
    seeds = sample_cells(ctx, cq, prog, x0, y0, cells)
    print('seed count:', len(seeds))

    if not(seeds):
        return

    suffix = '{}-{}_{}'.format(
        to_unit(SAMPLES),
        to_unit(MIN_ITERS_SAMPLES),
        to_unit(MAX_ITERS_SAMPLES),
    )
    seed_name = f'seeds-{suffix}-{int(time.time())}.json'
    save_seeds(seeds, seed_name)
    print(f'saved seeds to "{seed_name}"')

def flame_palette():
    def f(x):
        r = min(255, 3 * x)
        g = min(255, max(0, 3 * x - r))
        b = min(255, max(0, 3 * x - r - g))
        return [r, g, b]
    return np.array(list(map(f, range(256))), dtype=np.uint8)

def render(seeds, img_name):
    ctx, cq, prog = cl_init()

    # compute per-pixel counts of orbits
    counts = render_seeds(ctx, cq, prog, seeds)

    palette = flame_palette()
    scaled = np.uint16(np.sqrt(counts / np.max(counts)) * (len(palette) - 1))
    image = palette[scaled]

    img = PIL.Image.fromarray(image, 'RGB')
    img = img.transpose(PIL.Image.Transpose.ROTATE_270)
    img.save(img_name)
    print(f'saved image "{img_name}"')

def animate(seeds, img_prefix):
    ctx, cq, prog = cl_init()
    palette = flame_palette()

    tot_frames = ANIMATE_SECONDS * ANIMATE_FPS
    max_iters = MIN_ITERS_SAMPLES
    step = max_iters // tot_frames
    iter_checkpoints = list(range(step, max_iters+step, step))
    prev = None

    for i, counts in enumerate(render_seeds_gen(ctx, cq, prog,
                                                seeds, iter_checkpoints),
                               start=1):
        if prev is None:
            prev = np.zeros_like(counts)
        diff = counts - prev
        prev = counts

        # combine total counts with difference to last counts
        # for more visually "active" images
        maxi = np.maximum(counts / max(1, np.max(counts)),
                          diff / max(1, np.max(diff)))
        scaled = np.uint16(np.sqrt(maxi) * (len(palette) - 1))
        image = palette[scaled]

        img = PIL.Image.fromarray(image, 'RGB')
        img = img.transpose(PIL.Image.Transpose.ROTATE_270)
        img_name = f'{img_prefix}-{i:05}.png'
        img.save(img_name)
        print(f'saved image "{img_name}" - {i} / {len(iter_checkpoints)}')

def do_load_seeds(fnames):
    seeds = []
    for fname in fnames:
        seeds.extend(load_seeds(fname))
    if not seeds:
        print('error: no input')
        sys.exit(1)
    print('seed count:', len(seeds))
    return seeds

def do_render(args):
    if args.output is None:
        if len(args.seeds) == 1:
            [fname] = args.seeds
            prefix = 'seeds-'
            suffix = '.json'
            if fname.startswith(prefix) and fname.endswith(suffix):
                middle = fname[len(prefix):-len(suffix)]
                args.output = f'bbrot-{middle}.png'
        if args.output is None:
            args.output = f'bbrot-{int(time.time())}.png'

    seeds = do_load_seeds(args.seeds)
    render(seeds, args.output)

def do_animate(args):
    seeds = do_load_seeds(args.seeds)
    animate(seeds, args.prefix)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand')

    cmd_compute = subparsers.add_parser('compute')

    cmd_render = subparsers.add_parser('render')
    cmd_render.add_argument('--output', '-o',
                            help='name of image output file')
    cmd_render.add_argument('seeds',
                            help='name of seed input file(s)',
                            nargs='+')

    cmd_animate = subparsers.add_parser('animate')
    cmd_animate.add_argument('--prefix', '-p',
                             help='prefix of image output files',
                             required=True)
    cmd_animate.add_argument('seeds',
                             help='name of seed input file(s)',
                             nargs='+')

    args = parser.parse_args(sys.argv[1:])

    if args.subcommand is None:
        print('error: a subcommand is required')
        sys.exit(1)

    if args.subcommand == 'compute':
        compute()

    elif args.subcommand == 'render':
        do_render(args)

    elif args.subcommand == 'animate':
        do_animate(args)

if __name__ == '__main__':
    main()
