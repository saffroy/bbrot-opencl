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

__kernel void mandel_trace(int max_iters,
                           __global int *seed_list_d,
                           __global FLOAT *x0_d,
                           __global FLOAT *y0_d,
                           __global FLOAT *x_d,
                           __global FLOAT *y_d,
                           __global int *buff_d,
                           __global int *iters_d,
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
        int iters = iters_d[seed];

        FLOAT x2 = x * x;
        FLOAT y2 = y * y;

        int n = 0;

        while ((x2 + y2 < 4.0)
               && (n < MAX_LOOPS)
               && ((max_iters < 0)
                   || (iters < max_iters)))
        {
                y = 2 * x * y + y0;
                x = x2 - y2 + x0;

                n++;
                iters++;
                inc_pix(x, y, buff);

                x2 = x * x;
                y2 = y * y;
        }

        x_d[seed] = x;
        y_d[seed] = y;
        iters_d[seed] = iters;
        done_d[seed] = (x2 + y2 >= 4.0)
                || (max_iters >= 0 && iters >= max_iters);
}
