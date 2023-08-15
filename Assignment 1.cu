/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-1
 * Description: Computation of a matrix C = Kronecker_prod(A, B.T)
 *              where A and B are matrices of dimension (m, n) and
 *              the output is of the dimension (m * n, m * n). 
 * Note: All lines marked in --> should be replaced with code. 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>   // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
using namespace std;

ofstream outfile; // The handle for printing the output

__global__ void per_row_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....

    //we are assuming to calculate the values by assuming the A matrix as blocks and B matrix as threads per block
    // so we are calclating the blockid and threadid for coorecsponding block here in ROW wise manner and multiplying
    // correctly to get the C value and findingthe C index by that id and putting in right place

    long int id = blockIdx.x;
    long int id2 = threadIdx.x;
    for(long int t1=0;t1<n;t1++){
        for(long int t2=0;t2<n;t2++){
            long int c_index = id*m*n*n+t2*m*n+id2+t1*m;
            long int a_index = id*n+t1;
            long int b_index = id2*n+t2;
            C[c_index] = A[a_index]*B[b_index];
        }
    }
}

__global__ void per_column_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....

//we are assuming to calculate the values by assuming the A matrix as blocks and B matrix as threads per block
    // so we are calclating the blockid and threadid for coorecsponding block here in COLUMN wise manner and multiplying
    // correctly to get the C value and findingthe C index by that id and putting in right place

    long int id = blockIdx.x;
    long int id2 = threadIdx.x;
    for(long int t1=0;t1<m;t1++){
        for(long int t2=0;t2<m;t2++){
            long int c_index = id*m+t2+id2*m*n+t1*m*n*n;
            long int a_index = id+t1*n;
            long int b_index = id2+t2*n;
            C[c_index] = A[a_index]*B[b_index];
        }
    }
}

__global__ void per_element_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....
    // long int id = blockIdx.x*blockDim.x+threadIdx.x;
    long long id1 = blockIdx.x*gridDim.y+blockIdx.y;
    long int id2 = id1*blockDim.x*blockDim.y;
    long int id3 = id2+threadIdx.x*blockDim.y+threadIdx.y;

    if(id3 < m*n*m*n){
        // printf("%3d\n",id3);

       long int row = id3/(m*n);
       long int col = id3%(m*n);

        long int a_row = row/n;
        long int a_col = col/m;
        long int b_row = col%m;
        long int b_col = row%n;

        long int a_index = a_row*n+a_col;
        long int b_index = b_row*n+b_col;

        C[id3] = A[a_index]*B[b_index];

    }
}

/**
 * Prints any 1D array in the form of a matrix
 **/
void printMatrix(long int *arr, long int rows, long int cols, char* filename){
    outfile.open(filename);
    for(long int i = 0; i < rows; i++){
        for(long int j = 0; j < cols; j++){
            outfile<<arr[i * cols + j]<<" ";
        }
        outfile<<"\n";
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    long int m,n;	
    cin>>m>>n;	

    // Host_arrays 
    long int *h_a,*h_b,*h_c;

    // Device arrays 
    long int *d_a,*d_b,*d_c;
	
    // Allocating space for the host_arrays 
    h_a = (long int *) malloc(m * n * sizeof(long int));
    h_b = (long int *) malloc(m * n * sizeof(long int));	
    h_c = (long int *) malloc(m * m * n * n * sizeof(long int));	

    // Allocating memory for the device arrays 

    // --> Allocate memory for A on device 
    cudaMalloc(&d_a,m*n*sizeof(long int));
    // --> Allocate memory for B on device 
    cudaMalloc(&d_b,m*n*sizeof(long int));
    // --> Allocate memory for C on device 
    cudaMalloc(&d_c,m*m*n*n*sizeof(long int));

    // Read the input matrix A 
    for(long int i = 0; i < m * n; i++) {
        cin>>h_a[i];
    }

    //Read the input matrix B 
    for(long int i = 0; i < m * n; i++) {
        cin>>h_b[i];
    }

    // Transfer the input host arrays to the device 
    // --> Copy A from Host to Device
    cudaMemcpy(d_a,h_a,m*n*sizeof(long int),cudaMemcpyHostToDevice);
    // --> Copy B from Host to Device 
    cudaMemcpy(d_b,h_b,m*n*sizeof(long int),cudaMemcpyHostToDevice);

    long int gridDimx, gridDimy;
    
    // Launch the kernels
    /**
     * Kernel 1 - per_row_AB_kernel
     * To be launched with 1D grid, 1D block
     * Each thread should process a complete row of A, B
     **/

    // --> Set the launch configuration 

    double starttime = rtclock();  

    // --> Launch the kernel 
    per_row_AB_kernel<<<m,m>>>(d_a,d_b,d_c,m,n);
    cudaDeviceSynchronize();                                                           

    double endtime = rtclock(); 
	  printtime("GPU Kernel-1 time: ", starttime, endtime);  

    // --> Copy C from Device to Host 
    cudaMemcpy(h_c,d_c,m*m*n*n*sizeof(long int),cudaMemcpyDeviceToHost);
    

    printMatrix(h_c, m * n, m * n,"kernel1.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(int));

    /**
     * Kernel 2 - per_column_AB_kernel
     * To be launched with 1D grid, 2D block
     * Each thread should process a complete column of  A, B
     **/
    
    // --> Set the launch configuration
    dim3 block2(n,2,1);

    starttime = rtclock(); 

    // --> Launch the kernel 
    per_column_AB_kernel<<<n,block2>>>(d_a,d_b,d_c,m,n);
    cudaDeviceSynchronize(); 

    endtime = rtclock(); 
  	printtime("GPU Kernel-2 time: ", starttime, endtime);  

    // --> Copy C from Device to Host
    cudaMemcpy(h_c,d_c,m*m*n*n*sizeof(long int),cudaMemcpyDeviceToHost);

    printMatrix(h_c, m * n, m * n,"kernel2.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(int));

    /**
     * Kernel 3 - per_element_kernel
     * To be launched with 2D grid, 2D block
     * Each thread should process one element of the output 
     **/
    gridDimx = ceil(float(n * n) / 16);
    gridDimy = ceil(float(m * m) / 64);
    dim3 grid3(gridDimx,gridDimy,1);
    dim3 block3(64,16,1);

    starttime = rtclock();  

    // --> Launch the kernel 
    per_element_kernel<<<grid3,block3>>>(d_a,d_b,d_c,m,n);
    cudaDeviceSynchronize();                                                              

    endtime = rtclock();  
	printtime("GPU Kernel-3 time: ", starttime, endtime);  

    // --> Copy C from Device to Host
    cudaMemcpy(h_c,d_c,m*m*n*n*sizeof(long int),cudaMemcpyDeviceToHost);
    printMatrix(h_c, m * n, m * n,"kernel3.txt");

    return 0;
}

