#ifndef KMEANS_CUDA_PROJECT_PLOT_ADV_H
#define KMEANS_CUDA_PROJECT_PLOT_ADV_H

#include "matplotlibcpp.h"
#include "Record.h"
#include <string>
#include <list>
#include <vector>
#include <map>
#include <cmath>
#include <string>

void plot_adv_kmeans(float* centroids_x,float* centroids_y,float* records_x,float* records_y, int* assigment, int num_clus,int num_rec, std::string s){

    namespace plt = matplotlibcpp;
    plt::figure_size(1200, 780);

    std::vector<std::vector<double>> x_positions(num_clus);
    std::vector<std::vector<double>> y_positions(num_clus);
    for(int i=0; i< num_rec; i++){
        x_positions[assigment[i]].push_back(records_x[i]);
        y_positions[assigment[i]].push_back(records_y[i]);
    }
    for(int i=0; i< num_clus; i++){
        plt::plot(x_positions[i], y_positions[i], "o");
    }
    for(int i=0; i< num_clus; i++){
        std::vector<double> x_positions_c;
        std::vector<double> y_positions_c;

        x_positions_c.push_back(centroids_x[i]);
        y_positions_c.push_back(centroids_y[i]);
        plt::plot(x_positions_c, y_positions_c, "k^");
    }
    plt::title(s);
    plt::show();
}

using namespace std;

void show_clusters(map<Record, list<Record>> clusters, string s){

    namespace plt = matplotlibcpp;

    vector<double> x_positions;
    vector<double> y_positions;
    plt::figure_size(1200, 780);
    int k = 0;
    for (auto cluster: clusters){
        for(auto record: cluster.second){
            x_positions.push_back(record.getPx());
            y_positions.push_back(record.getPy());
        }
        plt::plot(x_positions, y_positions, "o");
        vector<double> x = {cluster.first.getPx()};
        vector<double> y = {cluster.first.getPy()};

        plt::plot(x, y, "k^");
        k++;
        x_positions.clear();
        y_positions.clear();
    }
    plt::title(s);
    plt::show();

}

#endif //KMEANS_CUDA_PROJECT_PLOT_ADV_H
