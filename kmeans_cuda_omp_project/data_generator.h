#ifndef KMEANS_CUDA_OMP_PROJECT_DATA_GENERATOR_H
#define KMEANS_CUDA_OMP_PROJECT_DATA_GENERATOR_H

#include <random>
#include <iostream>
#include <chrono>

void generate_gaussian_list_records(int num_record, float min, float max, int k, int deviation){

    FILE *fptr;
    if ((fptr = fopen("/home/lorenzo/CLionProjects/kmeans_cuda_omp_project/data_generated","w")) == NULL){
        printf("Error! opening file");
        exit(1);
    }

    int rec_x_clus = int(num_record / k);
    int resto = num_record % k;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::mt19937 e2(seed);
    //std::cout<<"seed: "<<seed<<std::endl;
    //std::cout<<"Centers are: "<<std::endl;

    for (int i = 0; i<k; i++) {

        std::uniform_real_distribution<double> unif(min,max);
        double center_x = unif(e2);
        double center_y = unif(e2);
        //std::cout<<"x: "<<center_x<<" y: "<<center_y<<std::endl;

        std::normal_distribution<double> distribution_x(center_x,deviation);
        std::normal_distribution<double> distribution_y(center_y,deviation);

        for (int j = 0; j<rec_x_clus; j++) {
            double number_x = distribution_x(generator);
            double number_y = distribution_y(generator);

            fprintf(fptr,"%d %f %f\n",i*rec_x_clus+j, number_x, number_y);
        }
        if(i==k-1){
            for (int j = 0; j<resto; j++) {
                double number_x = distribution_x(generator);
                double number_y = distribution_y(generator);

                fprintf(fptr,"%d %f %f\n",i*rec_x_clus+j, number_x, number_y);
            }
        }
    }
    fclose(fptr);
}
void choose_centroids_from_records(int num_record, int k){
    FILE *fptrW;
    if ((fptrW = fopen("/home/lorenzo/CLionProjects/kmeans_cuda_omp_project/data_centroids","w")) == NULL){
        printf("Error! opening file");
        exit(1);
    }
    FILE *fptrR;
    if ((fptrR = fopen("/home/lorenzo/CLionProjects/kmeans_cuda_omp_project/data_generated","r")) == NULL){
        printf("Error! opening file");
        exit(1);
    }
    std::uniform_int_distribution<int> distribution(0,num_record);
    std::mt19937 e2(std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<int> vector_index={};
    for (int i=0; i< k; i++){
        int id = distribution(e2);
        vector_index.push_back(id);
        //std::cout<<id<<std::endl;
    }
    sort(vector_index.begin(), vector_index.end());
    int i=0;
    int j=0;
    while(i<k){
        int id;
        float x;
        float y;
        while(j!=vector_index[i]){
            fscanf(fptrR,"%d %f %f\n",&id, &x, &y);
            j++;
        }
        fscanf(fptrR,"%d %f %f\n", &id, &x, &y);
        fprintf(fptrW,"%d %f %f\n",i, x, y);
        i++;
    }
    fclose(fptrR);
    fclose(fptrW);
}


#endif //KMEANS_CUDA_OMP_PROJECT_DATA_GENERATOR_H
