#ifndef PROJECT_KMEANS_KMEANS_PARALLEL_H
#define PROJECT_KMEANS_KMEANS_PARALLEL_H

#include <boost/lockfree/queue.hpp>
#include "Record.h"
#include <vector>
#include <list>
#include <float.h>
#include <chrono>
#include <map>

using namespace std;

map<Record, boost::lockfree::queue < Record>*> clusters;

vector<Record> get_records_from_file_o(int num_records){
    vector<Record> records;
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

vector<Record> get_centroids_from_file_o(int k){
    vector<Record> centroids;
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


void assign_to_cluster_p(Record record, Record centroid)
{
    auto my_cluster = clusters.find(centroid);
    my_cluster->second->push(record);
}

vector<Record> random_centroids_parallel(int k, vector<Record> records, unsigned int seed) {

    vector<Record> centroids;
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
vector<Record> relocate_centroids_parallel() {

    vector<Record> centroids_vector(clusters.size());
    vector<Record> centroid_vector_tmp;

    for(auto cluster:clusters){
        centroid_vector_tmp.push_back(cluster.first);
    }
    #pragma omp parallel for
    for (int i = 0; i < clusters.size(); i++){
        auto centroid = centroid_vector_tmp[i];
        auto cluster = clusters.find(centroid);

        Record r;
        double sumx = 0;
        double sumy = 0;
        int size = 0;

        while(cluster->second->pop(r)){
            size++;
            sumx += r.getPx();
            sumy += r.getPy();
        }
        //auto new_centroid = average_parallel2(cluster->first, list);
        sumx = sumx/size;
        sumy = sumy/size;

        centroids_vector[i] = Record(i, sumx, sumy);
    }
    return centroids_vector;
}

void init_clusters(vector<Record> centroids)
{
    for(auto centroid:centroids){
        auto* empty_list = new boost::lockfree::queue<Record>(0);
        clusters.insert({centroid, empty_list});
    }
}

void delete_clusters_p(){
    for(auto a: clusters){
        delete a.second;
    }
    clusters.clear();
}

map<Record, list<Record>> lastState_p(){
    map<Record, list<Record>> state;
    for(auto clust: clusters){
        list<Record> list;
        Record c(clust.first);
        while(!clust.second->empty()){
            Record r;
            clust.second->pop(r);
            list.push_back(r);
        }
        state.insert({c, list});
    }
    return state;
}

Record nearest_centroid_p(Record record, vector<Record> centroids)
{
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

bool close_centroids_parallel(vector<Record> old_centroids,vector<Record> new_centroids, float MIN_TOLLERANCE){
    if(old_centroids.size()==0){
        return false;
    }
    for (int i =0; i< old_centroids.size(); i++){
        if( record_distance(old_centroids[i], new_centroids[i])>MIN_TOLLERANCE){
            return false;
        }
    }
    return true;
}


map<Record, list<Record>>  fit_parallel(int k, int num_records, int maxIterations, float MIN_TOLLERANCE){
    vector<Record> records = get_records_from_file_o(num_records);
    vector<Record> centroids = get_centroids_from_file_o(k);
    vector<Record> old_centroids = {};

    for (int i = 0; i < maxIterations; i++) {
        init_clusters(centroids);
        bool isLastIteration = i == maxIterations - 1;

        #pragma omp parallel for
        for (int j=0; j<records.size(); j++) {
            Record centroid = nearest_centroid_p(records[j], centroids);
            assign_to_cluster_p(records[j], centroid);
        }

        if (isLastIteration or close_centroids_parallel(old_centroids,centroids,MIN_TOLLERANCE)) {
            //cout<<"End at iteration: "<<i<<endl;
            break;
        }
        old_centroids = centroids;
        centroids = relocate_centroids_parallel();
        delete_clusters_p();
    }

    map<Record, list<Record>> last_state = lastState_p();
    delete_clusters_p();
    return last_state;
}



#endif //PROJECT_KMEANS_KMEANS_PARALLEL_H
