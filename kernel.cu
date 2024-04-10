#define O_TILE_WIDTH 16
#define MAX_KERNEL_WIDTH 10

__constant__ float M[MAX_KERNEL_WIDTH * MAX_KERNEL_WIDTH];

__global__
void convolution2D(float* in, float* out, int width, int height, int channels, float* kernel, int kernel_width)
{   
    // output width, stride = 1
    //int ow = (width-(kernel_width-1)-1)/1+1;
    // output height, stide = 1
    //int oh = (height-(kernel_width-1)-1)/1+1;

    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    //int cha = blockIdx.z * blockDim.z + threadIdx.z;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    float sum = 0.0f;

    int row_start_point = ty - kernel_width/2;
    int col_start_point = tx - kernel_width/2;

    for(int i=0; i<kernel_width; i++){
        for(int j=0; j<kernel_width; j++){

            int input_row_point = row_start_point+i;
            int input_col_point = col_start_point+j;

            //checking for computation necessary or not?
            if(input_row_point>=0 && input_row_point<height && input_col_point>=0 && input_col_point <width){
                //sum = sum + kernel[i * kernel_width + j] * in[ 0 + input_row_point * width + input_col_point];
                sum = sum + kernel[i * kernel_width + j] * in[ bz*height*width + input_row_point * width + input_col_point];
            }
            //if(tx ==0 && ty==0)
            //printf("kernel %f input %f sum %f \n", kernel[i * kernel_width + j], in[0 + input_row_point * width + input_col_point], sum);
        }

    }

    // 0 for 1 channel only
    //out[0+ ty * width+ tx] = sum;
    out[bz*height*width + ty * width+ tx] = sum;

}


__global__
void convolution2D_sharedmem(float* in, float* out, int width, int height, int channels, float* kernel, int kernel_width)
{

}
