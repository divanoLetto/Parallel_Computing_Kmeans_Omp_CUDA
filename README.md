# Parallel_Computing_Kmeans_Omp_CUDA

## Abstract 
This paper is focused on the analysis of the **Kmeans clustering algorithm** and on the study of its performance.
In particular, in addition to the sequential version, other versions have been created that exploit the parallelism of processors and **GPUs** through the programming languages **Omp** and **CUDA**.

## Results
The Kmeans clustering algorithm's performance has been improved by **multi-processors parallelism** obteined through the **Omp** framework which allowed to obtain a speedup close to the number of processors avaible. Subsequently, through the parallelism of the **GPUs**, we develope a version of the algorithm by the parallel computing platform **CUDA** that has enormously higher speedups, in the order of hundreds, compared to both the sequential and the Omp parallel version.
