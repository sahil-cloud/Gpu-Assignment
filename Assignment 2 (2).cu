
#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

__global__ void transpose_kernel(int *d_matrix1,int *d_matrix1T,int rows,int cols){
  //instead of multipliying the row colo wise we are taking the transpose so that we can
  //use of coalesed memory and better the performance by taking whole row together
    int row = blockIdx.x;
    int col = threadIdx.x;

    int id = row * cols + col;
    int id2 = col * rows + row;
    d_matrix1T[id2] = d_matrix1[id];
}

__global__ void multiply_kernel(int *d_matrixResult,int *d_matrix1,int *d_matrix2,int p,int q,int r){
    
    //dynamic allocated shared memory and three pointers diveded into q size each
    //taking whole row of both the matrices into the shared memory and computing their multiplication
    //the result also store in the shared memory and then we sum up all the values to get the
    // sum and the element for the computation

    extern __shared__ int temp[];
    int *temp1 = temp;
    int *temp2 = temp+q;
    int *temp3 = temp2+q;
    
    int row = blockIdx.x; // p dimenson
    int col = blockIdx.y; // r dimension

    int sum = 0;
    d_matrixResult[row*r+col] = 0;

    int id = threadIdx.x;

    temp1[id] = d_matrix1[q*row+id];
    temp2[id] = d_matrix2[q*col+id];

    temp3[id] = temp1[id]*temp2[id];

    __syncthreads();

      for(int i=0;i<q;i++){
        sum += temp3[i];
      }

      d_matrixResult[row*r+col] = sum;

}

__global__ void addition_kernel(int *d_matrixAddition,int *d_matrixMult1,int *d_matrixMult2,int p,int r){
    
    //here we are taking the shared memory for addition of two variables 
    //we always launch 1 thread so that id is 0 always we claculate sum each element by element
    //so we take whole row into the shared memory so that we can be better coalesed and also
    // access the memory an compute the sum one by one elemnt

    extern __shared__ int temp[];

    int *temp1 = temp;
    int *temp2 = temp+r;

    int row = blockIdx.x; //p
    int col = blockIdx.y; // r

    int id = threadIdx.x;
    d_matrixAddition[row*r+col] = 0;

     temp1[id] = d_matrixMult1[r*row+col];
    temp2[id] = d_matrixMult2[r*row+col];

    __syncthreads();
    
       d_matrixAddition[row*r+col] = temp1[id] + temp2[id];

}




// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */

  //transpose kernel
  int *d_matrixBT;
  cudaMalloc(&d_matrixBT,q*r*sizeof(int));

  //Instead of taking the row-col wise multiplication we are taking transpose so that
  // we can use memory-coalesing so that access to the element could be faster and better
  
  transpose_kernel<<<q,r>>>(d_matrixB,d_matrixBT,q,r);
  cudaDeviceSynchronize();


  // ----------------------------------------------------------------------
  int *h_matrixBT;
  h_matrixBT = (int*) malloc(r * q * sizeof(int));
  cudaMemcpy(h_matrixBT, d_matrixBT, r * q * sizeof(int), cudaMemcpyDeviceToHost);

  //multiply kernel
  int *d_matrixMult1,*d_matrixMult2;
  dim3 grid(p,r);
    //allocate 3*q* size to the dnamic shared memory
  cudaMalloc(&d_matrixMult1,p*r*sizeof(int));
  cudaMalloc(&d_matrixMult2,p*r*sizeof(int));
  multiply_kernel<<<grid,q,3*q*sizeof(int)>>>(d_matrixMult1,d_matrixA,d_matrixBT,p,q,r);
  cudaDeviceSynchronize();
  multiply_kernel<<<grid,q,3*q*sizeof(int)>>>(d_matrixMult2,d_matrixC,d_matrixD,p,q,r);
  cudaDeviceSynchronize();

  //addition kernel
  dim3 grid2(p,r);
  //allocate 3*r* size to the dnamic shared memory
  addition_kernel<<<grid2,1,3*r*sizeof(int)>>>(d_matrixE,d_matrixMult1,d_matrixMult2,p,r);
  cudaDeviceSynchronize();

	/* Configure and launch kernels */

	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

//function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	

