// -*- c -*-

#include "bbrot-generated.h"

#ifndef FLOAT
#define FLOAT double
#endif

__kernel void mandel_iters(int max_iters,
                           __global FLOAT *x0_d,
                           __global FLOAT *y0_d,
                           __global FLOAT *x_d,
                           __global FLOAT *y_d,
                           __global int *iters_d,
                           __global int *done_d)
{
        int rank = get_global_id(0);

        if (done_d[rank])
                return;

        FLOAT x0 = x0_d[rank];
        FLOAT y0 = y0_d[rank];
        FLOAT x = x_d[rank];
        FLOAT y = y_d[rank];
        int iters = iters_d[rank];

        FLOAT x2 = x * x;
        FLOAT y2 = y * y;

        int n = 0;

        while ((x2 + y2 < 4.0)
               && (n < MAX_LOOPS)
               && (iters < max_iters))
        {
                y = 2 * x * y + y0;
                x = x2 - y2 + x0;

                n++;
                iters++;

                x2 = x * x;
                y2 = y * y;
        }

        x_d[rank] = x;
        y_d[rank] = y;
        iters_d[rank] = iters;
        done_d[rank] = (x2 + y2 >= 4.0) || iters >= max_iters;
}

static void inc_pix(FLOAT x, FLOAT y, __global int *buff)
{
        int xi = (x - XMIN) / XRANGE * STEPS;
        int yi = (y - YMIN) / YRANGE * STEPS;

        if ((xi >= 0) && (xi < STEPS)
            && (yi >= 0) && (yi < STEPS)) {
                int off = xi + yi*STEPS;
                buff[off]++;
        }
}

__kernel void mandel_trace(__global int *seed_list_d,
                           __global FLOAT *x0_d,
                           __global FLOAT *y0_d,
                           __global FLOAT *x_d,
                           __global FLOAT *y_d,
                           __global int *buff_d,
                           __global int *done_d)
{
        int rank = get_global_id(0);
        int seed = seed_list_d[rank];

        if (done_d[seed])
                return;

        __global int *buff = buff_d + rank * (STEPS * STEPS);

        FLOAT x0 = x0_d[seed];
        FLOAT y0 = y0_d[seed];
        FLOAT x = x_d[seed];
        FLOAT y = y_d[seed];

        FLOAT x2 = x * x;
        FLOAT y2 = y * y;

        int n = 0;

        while ((x2 + y2 < 4.0)
               && (n < MAX_LOOPS))
        {
                y = 2 * x * y + y0;
                x = x2 - y2 + x0;

                n++;
                inc_pix(x, y, buff);

                x2 = x * x;
                y2 = y * y;
        }

        x_d[seed] = x;
        y_d[seed] = y;
        done_d[seed] = (x2 + y2 >= 4.0);
}

__kernel void counts_to_image(int max_count,
                              int palette_len,
                              __global char *palette_d,
                              __global int *counts_d,
                              __global char *image_d)
{
        int rank = get_global_id(0);

        float ratio = (float)counts_d[rank] / (float)max_count;
        int idx = min(palette_len, (int)(sqrt(ratio) * palette_len));

        __global char *pal = palette_d + 3*idx;
        __global char *pix = image_d + 3*rank;
        pix[0] = pal[0];
        pix[1] = pal[1];
        pix[2] = pal[2];
}

static int in_mandel(int max_iters,
                     int i,
                     int j,
                     __global int *iters_d)
{
        int off = i + j*STEPS;
        return iters_d[off] == max_iters ? 1 : 0;
}

__kernel void gen_cell_list(int max_iters,
                            __global int *iters_d,
                            __global int *cells_d,
                            __global int *cell_count_d)
{
        __global int *plist = cells_d;
        int num_cells = 0;

        for (int j = 0; j < STEPS-1; j++) {
                for (int i = 0; i < STEPS-1; i++) {
                        int s = 0;
                        s += in_mandel(max_iters, i+0, j+0, iters_d);
                        s += in_mandel(max_iters, i+1, j+0, iters_d);
                        s += in_mandel(max_iters, i+0, j+1, iters_d);
                        s += in_mandel(max_iters, i+1, j+1, iters_d);

                        // Eligible cell has some corners inside M-set
                        // and some outside.
                        if (s > 0 && s < 4) {
                                (*plist++) = i;
                                (*plist++) = j;
                                num_cells++;
                        }
                }
        }

        *cell_count_d = num_cells;
}
