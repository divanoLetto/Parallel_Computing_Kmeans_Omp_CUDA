#ifndef PROJECT_KMEANS_KMEANS_SEQUENTIAL_H
#define PROJECT_KMEANS_KMEANS_SEQUENTIAL_H

#include "Record.h"
#include <list>
#include <map>
#include <algorithm>
#include <float.h>
#include <iostream>
#include <limits>
#include <random>

using namespace std;

list<Record> random_centroids_from_records(list<Record> records, int k, unsigned int seed) {

    list<Record> centroids;
    int maxn = records.size();
    int index;
    list<int> previus_index;
    srand(seed);

    for (int i=0; i<k;i++) {
        bool found;
        do {
            index = (rand() % (maxn));
            found = (find(previus_index.begin(), previus_index.end(), index) != previus_index.end());
        } while (found);
        auto it = records.begin();
        previus_index.push_back(index);
        advance(it, index);
        centroids.push_back(Record(*it));
    }
    int i=0;
    for(auto it = centroids.begin();  it != centroids.end(); it++){
        it->setId(i++);
    }
    return centroids;
}

list<Record> random_centroids_from_distribution(list<Record> records, int k, unsigned int seed) {

    //std::cout<<"Centroids are: "<<std::endl;
    float min = std::numeric_limits<float>::infinity();
    float max = 0;
    for( auto r: records){
        if (max<r.getPx()){
            max=r.getPx();
        }
        if (max<r.getPy()){
            max=r.getPy();
        }
        if (min>r.getPx()){
            min=r.getPx();
        }
        if (min>r.getPy()){
            min=r.getPy();
        }
    }
    list<Record> centroids;
    std::uniform_real_distribution<double> unif(min,max);
    std::random_device rd;
    std::mt19937 e2(rd());
    for (int i = 0; i<k; i++){
        double center_x = unif(e2);
        double center_y = unif(e2);
        centroids.push_back(Record(i, center_x, center_y));
        //std::cout<<"x: "<<center_x<<" y: "<<center_y<<std::endl;
    }
    return centroids;
}

list<Record> get_records_from_file_s(int num_records){
    list<Record> records;
    FILE *fptr;
    if ((fptr = fopen("/home/lorenzo/CLionProjects/kmeans_cuda_omp_project/data_generated","r")) == NULL){
        printf("Error! opening file");
        exit(1);
    }
    for(int i=0; i<num_records; i++){
        int id;
        float number_x;
        float number_y;
        fscanf(fptr,"%d %f %f\n", &id, &number_x, &number_y);
        records.emplace_back(id,number_x,number_y);
    }
    fclose(fptr);
    return records;
}

list<Record> get_centroids_from_file_s(int k){
    list<Record> centroids;
    FILE *fptr;
    if ((fptr = fopen("/home/lorenzo/CLionProjects/kmeans_cuda_omp_project/data_centroids","r")) == NULL){
        printf("Error! opening file");
        exit(1);
    }
    for(int i=0; i<k; i++) {
        int id;
        float number_x;
        float number_y;
        fscanf(fptr, "%d %f %f\n", &id, &number_x, &number_y);
        centroids.emplace_back(id, number_x, number_y);
    }
    fclose(fptr);
    return centroids;
}

Record nearestCentroid(Record record, list<Record> centroids) {

    double minimumDistance = DBL_MAX;
    Record nearest;
    for (Record centroid : centroids) {
        double currentDistance = record_distance(record, centroid);

        if (currentDistance < minimumDistance) {
            minimumDistance = currentDistance;
            nearest = centroid;
        }
    }
    return nearest;
}

void assignToCluster(map<Record, list<Record>>* clusters,  Record record, Record centroid) {

    auto my_cluster = clusters->find(centroid);
    if ( my_cluster == clusters->end() ) {
        list<Record> empty_list={};
        clusters->insert({centroid, empty_list});
        my_cluster = clusters->find(centroid);
    }
    my_cluster->second.push_back(record);
}

Record average(Record centroid, list<Record> records, int nk)
{
    if (records.empty()) {
        return centroid;
    }
    double px = 0;
    double py = 0;
    for (Record record : records) {
        px += record.getPx();
        py += record.getPy();
    }
    px = px/records.size();
    py = py/records.size();
    return Record(nk, px, py);
}

list<Record> relocateCentroids(map<Record, list<Record>> clusters) {
    list<Record> centroids_list;
    int nk = 0;
    for (auto cluster: clusters){
        auto centroid = average(cluster.first, cluster.second, nk++);
        centroids_list.push_back(centroid);
    }
    return centroids_list;
}

bool close_centroids(list<Record> centroids_old,list<Record> centroids_new, float MIN_TOLLERANCE){
    if(centroids_old.size()==0){
        return false;
    }
    for(auto elem_old: centroids_old){
        for(auto elem_new: centroids_new){
            if(elem_new.getId()==elem_old.getId()){
                if(record_distance(elem_new, elem_old)>MIN_TOLLERANCE){
                    return false;
                }
            }
        }
    }
    return true;
}

map<Record, list<Record>> fit_sequential(int k, int num_records, int maxIterations, float MIN_TOLLERANCE){

    map<Record, list<Record>> clusters;
    map<Record, list<Record>> lastState;
    list<Record> records = get_records_from_file_s(num_records);
    list<Record> centroids = get_centroids_from_file_s(k);
    list<Record> old_centroids = {};

    for (int i = 0; i < maxIterations; i++) {
        bool isLastIteration = i == maxIterations - 1;

        for (Record record : records) {
            Record centroid = nearestCentroid(record, centroids);
            assignToCluster(&clusters, record, centroid);
        }
        //show_clusters(clusters, "k-means");
        if (isLastIteration or close_centroids(old_centroids,centroids, MIN_TOLLERANCE)) {
            //cout<<"End at iteration: "<<i<<endl;
            lastState = clusters;
            break;
        }
        old_centroids = centroids;
        centroids = relocateCentroids(clusters);
        clusters.clear();
    }
    return lastState;
}


#endif //PROJECT_KMEANS_KMEANS_SEQUENTIAL_H
