#include "kmeans_adv.h"
#include "plot_adv.h"
#include "data_generator.h"
#include "Record.h"
#include <list>
#include "kmeans_sequential.h"
#include "kmeans_omp.h"

#define DEVIATION 20.0

int main()
{
    float min = 0;
    float max = 700;
    bool iamhere = true;
    int num_experiments = 1;
    float min_tollerance = 0.01;
    int max_iteration = 100;
    int num_records = 100000;
    int num_cluster = 6;
    int TPB = 512;
    std::vector<int> choises = {10000};

    for(auto chooise : choises) {
        num_records = chooise;
        std::cout << "Stating test with N= " << num_records << " K= "<< num_cluster << " TPB= " <<TPB<<std::endl;

        std::vector<int> time_sequential = {};
        std::vector<int> time_omp = {};
        std::vector<int> time_cuda_slow = {};
        std::vector<int> time_cuda = {};

        //std::cout << "  iterazione:";
        for (int it = 0; it < num_experiments; it++) {

            float *host_centroids_x;
            float *host_centroids_y;
            float *host_records_x;
            float *host_records_y;
            int *assignment;
            int *n_cluster = new int();
            int *n_record = new int();
            float *host_centroids_x_slow;
            float *host_centroids_y_slow;
            float *host_records_x_slow;
            float *host_records_y_slow;
            int *assignment_slow;
            int *n_cluster_slow = new int();
            int *n_record_slow = new int();
            std::map<Record, std::list<Record>> clusters_sequential;
            std::map<Record, std::list<Record>> clusters_parallel;

            generate_gaussian_list_records(num_records, min, max, num_cluster, DEVIATION);
            choose_centroids_from_records(num_records, num_cluster);

            //std::cout << " " << it;

            //std::cout << "s";
            std::chrono::steady_clock::time_point begin_sequential = std::chrono::steady_clock::now();
            clusters_sequential = fit_sequential(num_cluster, num_records, max_iteration, min_tollerance);
            std::chrono::steady_clock::time_point end_sequential = std::chrono::steady_clock::now();

            //std::cout << "o";
            std::chrono::steady_clock::time_point begin_omp = std::chrono::steady_clock::now();
            clusters_parallel = fit_parallel(num_cluster, num_records, max_iteration, min_tollerance);
            std::chrono::steady_clock::time_point end_omp = std::chrono::steady_clock::now();

            //std::cout << "cs";
            std::chrono::steady_clock::time_point begin_cuda_slow = std::chrono::steady_clock::now();
            kmeans_cuda_slow(&host_centroids_x_slow, &host_centroids_y_slow, &host_records_x_slow, &host_records_y_slow,
                             &assignment_slow, n_cluster_slow, n_record_slow, num_records, num_cluster, max_iteration,
                             min_tollerance, TPB);
            std::chrono::steady_clock::time_point end_cuda_slow = std::chrono::steady_clock::now();

            //std::cout << "c";
            std::chrono::steady_clock::time_point begin_cuda = std::chrono::steady_clock::now();
            kmeans_cuda(&host_centroids_x, &host_centroids_y, &host_records_x, &host_records_y, &assignment, n_cluster,n_record, num_records, num_cluster, max_iteration, min_tollerance, TPB);
            std::chrono::steady_clock::time_point end_cuda = std::chrono::steady_clock::now();

            time_sequential.push_back(chrono::duration_cast<chrono::microseconds>(end_sequential - begin_sequential).count());
            time_omp.push_back(chrono::duration_cast<chrono::microseconds>(end_omp - begin_omp).count());
            time_cuda_slow.push_back(chrono::duration_cast<chrono::microseconds>(end_cuda_slow - begin_cuda_slow).count());
            time_cuda.push_back(chrono::duration_cast<chrono::microseconds>(end_cuda - begin_cuda).count());

            if (it == num_experiments - 1 and iamhere) {
                show_clusters(clusters_sequential, "sequential plot");
                show_clusters(clusters_parallel, "omp parallel plot");
                plot_adv_kmeans(host_centroids_x_slow, host_centroids_y_slow, host_records_x_slow, host_records_y_slow,
                                assignment_slow, *n_cluster_slow,
                                *n_record_slow, "cuda no shared m");
                plot_adv_kmeans(host_centroids_x, host_centroids_y, host_records_x, host_records_y, assignment,
                                *n_cluster,
                                *n_record, "cuda shared m");
            }
        }
        long int media_sequential = 0, media_omp = 0, media_cuda = 0, media_cuda_slow = 0;
        for (int j = 0; j < num_experiments; j++) {
            media_sequential += time_sequential[j];
            media_omp += time_omp[j];
            media_cuda += time_cuda[j];
            media_cuda_slow += time_cuda_slow[j];
        }
        media_sequential = media_sequential / num_experiments;
        media_omp = media_omp / num_experiments;
        media_cuda = media_cuda / num_experiments;
        media_cuda_slow = media_cuda_slow / num_experiments;
        float speedup_omp = (float) media_sequential / (float) media_omp;
        float speedup_cuda = (float) media_sequential / (float) media_cuda;
        float speedup_cuda_slow = (float) media_sequential / (float) media_cuda_slow;

        cout << "Time normal version         = " << media_sequential << " [µs]" << endl;
        cout << "Time omp parallel version   = " << media_omp << " [µs]" << endl;
        cout << "Time cuda parallel version without shared memory  = " << media_cuda_slow << " [µs]" << endl;
        cout << "Time cuda parallel version  with shared memory  = " << media_cuda << " [µs]" << endl;
        cout << "The omp speedup is : " << speedup_omp << endl;
        cout << "The cuda without shared memory speedup is : " << speedup_cuda_slow << endl;
        cout << "The cuda with shared memory speedup is : " << speedup_cuda << endl;
        cout << endl;

        if(num_experiments != 1 and iamhere){
            std::vector<int> v(num_experiments);
            std::iota(v.begin(), v.end(), 0);
            matplotlibcpp::plot(v, time_sequential, {{"label", "sequential"}});
            matplotlibcpp::plot(v, time_omp, {{"label", "omp"}});
            matplotlibcpp::plot(v, time_cuda, {{"label", "cuda shared m"}});
            matplotlibcpp::plot(v, time_cuda_slow, {{"label", "cuda no shared m"}});

            matplotlibcpp::xlabel("num exp");
            matplotlibcpp::ylabel("time");

            matplotlibcpp::legend();

            matplotlibcpp::show();
        }
    }
    return 0;
}
