#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <stdio.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value);

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
using namespace std;

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
    if (err == cudaSuccess)
        return;
    std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
    exit (1);
}

struct Record_adv {

    Record_adv(int size, std::vector<float>& x_vector_host, std::vector<float>& y_vector_host){
        size_t size_records = size * sizeof(float);
        CUDA_CHECK_RETURN(cudaMalloc(&x, size_records));
        CUDA_CHECK_RETURN(cudaMalloc(&y, size_records));
        CUDA_CHECK_RETURN(cudaMemcpy(x, x_vector_host.data(), size_records, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(y, y_vector_host.data(), size_records, cudaMemcpyHostToDevice));
    }
    Record_adv(int size){
        std::vector<float>x_vector_host(size, 0);
        std::vector<float>y_vector_host(size, 0);
        size_t size_records = size * sizeof(float);
        CUDA_CHECK_RETURN(cudaMalloc(&x, size_records));
        CUDA_CHECK_RETURN(cudaMalloc(&y, size_records));
        CUDA_CHECK_RETURN(cudaMemcpy(x, x_vector_host.data(), size_records, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(y, y_vector_host.data(), size_records, cudaMemcpyHostToDevice));
    }

    ~Record_adv() {
        cudaFree(x);
        cudaFree(y);
    }

    float * get_x(){
        return x;
    }
    float * get_y(){
        return x;
    }

    float * x{nullptr};
    float * y{nullptr};
};

__device__ float records_adv_distance_slow(float x_1, float y_1, float x_2, float y_2) {
  return sqrt((x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2));
}

Record_adv * get_records_from_file_slow(int NUM_RECORDS){

    std::vector<float> host_x;
    std::vector<float> host_y;
    FILE *fptr;
    if ((fptr = fopen("/home/lorenzo/CLionProjects/kmeans_cuda_omp_project/data_generated","r")) == NULL){
        printf("Error! opening file");
        exit(1);
    }
    for(int i=0; i<NUM_RECORDS; i++){
        int id;
        float number_x;
        float number_y;
        fscanf(fptr,"%d %f %f\n", &id, &number_x, &number_y);
        host_x.push_back(number_x);
        host_y.push_back(number_y);
    }
    fclose(fptr);
    Record_adv* records= new Record_adv(NUM_RECORDS, host_x, host_y);
    return records;
}

Record_adv * get_centroids_from_file_slow( int k){

    std::vector<float> host_x;
    std::vector<float> host_y;
    FILE *fptr;
    if ((fptr = fopen("/home/lorenzo/CLionProjects/kmeans_cuda_omp_project/data_centroids","r")) == NULL){
        printf("Error! opening file");
        exit(1);
    }
    for(int i=0; i<k; i++){
        int id;
        float number_x;
        float number_y;
        fscanf(fptr,"%d %f %f\n", &id, &number_x, &number_y);
        host_x.push_back(number_x);
        host_y.push_back(number_y);
    }
    Record_adv* records= new Record_adv(k, host_x, host_y);
    fclose(fptr);
    return records;

}

__global__ void  cluster_assigment_slow(float* records_x,float *records_y,float*centroids_x,float*centroids_y,float*sum_x,float*sum_y,int num_rec,int k,int * sizes_centroid_d, int* assignment_d){

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_rec) return;
    float x = records_x[index];
    float y = records_y[index];

    float best_distance = INFINITY;
    int best_cluster = -1;

    for (int cluster = 0; cluster < k; ++cluster) {
        const float distance = records_adv_distance_slow(x,y,centroids_x[cluster],centroids_y[cluster]);
        if (distance < best_distance) {
            best_distance = distance;
            best_cluster = cluster;
        }
    }
    assignment_d[index]= best_cluster;
    atomicAdd(&sum_x[best_cluster], x);
    atomicAdd(&sum_y[best_cluster], y);
    atomicAdd(&sizes_centroid_d[best_cluster], 1);
}

__global__ void relocations_centroids_slow(float * centroids_x, float* centroids_y, float* sum_x, float* sum_y, int* sizes_centroid_d){
    const int cluster = threadIdx.x;
    int size = max(sizes_centroid_d[cluster], 1);

    centroids_x[cluster] = sum_x[cluster] / size;
    centroids_y[cluster] = sum_y[cluster] / size;

    sizes_centroid_d[cluster] = 0;
    sum_x[cluster]=0;
    sum_y[cluster]=0;
}

__global__ void print_centroids_slow(float * centroids_x, float* centroids_y){
    const int cluster = threadIdx.x;
    if(cluster==0){
        printf("Print centroid\n");
    }
    __syncthreads();
    printf("Cluster id %d has x: %f y: %f\n", cluster, centroids_x[cluster], centroids_y[cluster]);
}

__global__ void print_records_slow(float * records_x, float* records_y){
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index==0){
        printf("Print records\n");
    }
    __syncthreads();
    printf("Record id %d has x: %f y: %f\n", index, records_x[index], records_y[index]);
}

bool close_centroids_cuda_slow(int j,float* old_centroids_x, float*  old_centroids_y, float*  old_centroids_tmp_x,float*  old_centroids_tmp_y,int NUM_CLUSTERS,float MIN_TOLLERANCE){
    if(j==0){
        return false;
    }
    for(int i=0; i<NUM_CLUSTERS; i++){
        float distance = sqrt(pow(old_centroids_x[i]- old_centroids_tmp_x[i],2)+pow(old_centroids_y[i]- old_centroids_tmp_y[i],2));
        if(distance>MIN_TOLLERANCE){
            return false;
        }
    }
    return true;
}

int kmeans_cuda_slow(float ** host_centroids_x, float ** host_centroids_y, float**host_records_x,float ** host_records_y, int**assignment, int* n_cluster, int *n_record,int NUM_RECORDS, int NUM_CLUSTERS, int MAX_ITERATIONS, float MIN_TOLLERANCE, int TPB)
{
    Record_adv * records = get_records_from_file_slow(NUM_RECORDS);
    Record_adv * centroids = get_centroids_from_file_slow(NUM_CLUSTERS);
    Record_adv * sum = new Record_adv(NUM_CLUSTERS);

    int* sizes_centroid_h = new int[NUM_CLUSTERS];
    for(int i=0; i<NUM_CLUSTERS;i++){
        sizes_centroid_h[i]=0;
    }
    int * sizes_centroid_d;
    int * assignment_d;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&sizes_centroid_d, NUM_CLUSTERS*sizeof(int)));
    CUDA_CHECK_RETURN(cudaMemcpy(sizes_centroid_d, sizes_centroid_h, NUM_CLUSTERS*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&assignment_d, NUM_RECORDS*sizeof(int)));

    cudaDeviceSynchronize();

    float * old_centroids_x = new float[NUM_CLUSTERS];
    float * old_centroids_y = new float[NUM_CLUSTERS];
    float * old_centroids_tmp_x = new float[NUM_CLUSTERS];
    float * old_centroids_tmp_y = new float[NUM_CLUSTERS];
    for(int y=0; y< NUM_CLUSTERS; y++){
        old_centroids_x[y]=0;
        old_centroids_y[y]=0;
    }

    for(int i=0; i<MAX_ITERATIONS; i++){

        cluster_assigment_slow<<<(NUM_RECORDS+TPB-1)/TPB,TPB>>>(records->x,
                records->y,
                centroids->x,
                centroids->y,
                sum->x,
                sum->y,
                NUM_RECORDS,
                NUM_CLUSTERS,
                sizes_centroid_d, assignment_d);
        cudaDeviceSynchronize();

        //controllo sullo spostamento dei centroidi: se sotto una soglia termina l'algoritmo
        CUDA_CHECK_RETURN(cudaMemcpy(old_centroids_tmp_x, centroids->x, NUM_CLUSTERS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(old_centroids_tmp_y, centroids->y, NUM_CLUSTERS * sizeof(float), cudaMemcpyDeviceToHost));
        if(close_centroids_cuda_slow(i,old_centroids_x, old_centroids_y, old_centroids_tmp_x, old_centroids_tmp_y,NUM_CLUSTERS, MIN_TOLLERANCE)) {
            break;
        }
        for (int j=0; j<NUM_CLUSTERS; j++){
            old_centroids_x[j]=old_centroids_tmp_x[j];
            old_centroids_y[j]=old_centroids_tmp_y[j];
        }


        relocations_centroids_slow<<<1, NUM_CLUSTERS>>>(centroids->x,
                centroids->y,
                sum->x,
                sum->y,
                sizes_centroid_d );

        cudaDeviceSynchronize();
    }

    *n_cluster = NUM_CLUSTERS;
    *n_record = NUM_RECORDS;
    *assignment = new int[NUM_RECORDS];
    *host_centroids_x = new float[NUM_CLUSTERS];
    *host_centroids_y = new float[NUM_CLUSTERS];

    CUDA_CHECK_RETURN(cudaMemcpy(*assignment, assignment_d, NUM_RECORDS* sizeof(int), cudaMemcpyDeviceToHost));

    size_t size_clusters = NUM_CLUSTERS * sizeof(float);
    CUDA_CHECK_RETURN(cudaMemcpy(*host_centroids_x, centroids->x, size_clusters, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(*host_centroids_y, centroids->y, size_clusters, cudaMemcpyDeviceToHost));

    *host_records_x= new float[NUM_RECORDS];
    *host_records_y= new float[NUM_RECORDS];

    size_t size_records = NUM_RECORDS * sizeof(float);
    CUDA_CHECK_RETURN(cudaMemcpy(*host_records_x, records->x, size_records, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(*host_records_y, records->y, size_records, cudaMemcpyDeviceToHost));
    records->~Record_adv();
	return 0;
}



