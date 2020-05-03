#ifndef KMEANS_CUDA_PROJECT_KMEANS_ADV_H
#define KMEANS_CUDA_PROJECT_KMEANS_ADV_H

int kmeans_cuda(float ** host_centroids_x, float ** host_centroids_y, float**host_records_x,float ** host_records_y,
        int **assignment, int* n_cluster, int *n_record, int NUM_RECORDS, int NUM_CLUSTERS,
        int MAX_ITERATIONS, float MIN_TOLLERANCE, int TPB);

int kmeans_cuda_slow(float ** host_centroids_x, float ** host_centroids_y, float**host_records_x,float ** host_records_y,
                int **assignment, int* n_cluster, int *n_record, int NUM_RECORDS, int NUM_CLUSTERS,
                int MAX_ITERATIONS, float MIN_TOLLERANCE, int TPB);


#endif //KMEANS_CUDA_PROJECT_KMEANS_ADV_H
