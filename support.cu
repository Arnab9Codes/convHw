/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void verify(float* in, float* out, float* kernel, int width, int height, int channels, int kernel_width)
{   
    //printf("\nverify---------\n");
    for (int ch = 0; ch < channels; ch++) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int row_start_point = row - kernel_width/2;
                int col_start_point = col - kernel_width/2;
                float val = 0.f;

                for (int i = 0; i < kernel_width; i++) {
                    for (int j = 0; j < kernel_width; j++) {
                        int row_idx = row_start_point + i;
                        int col_idx = col_start_point + j;

                        if (row_idx >= 0 && row_idx < height && col_idx >= 0 && col_idx < width) {
                            
                            val += in[ch*width*height + row_idx*width + col_idx]*kernel[i*kernel_width + j];
                            //printf(" in %f k %f\n", in[ch*width*height + row_idx*width + col_idx], kernel[i*kernel_width + j]);
                        }
                    }
                }

                out[ch*width*height + row*width + col] = val;
                //printf(" val %f\n",val);
            }
        }
    }
    //printf("\n verify -----------\n");
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}
