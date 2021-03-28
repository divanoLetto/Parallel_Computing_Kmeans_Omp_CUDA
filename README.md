# Parallel_Computing_Kmeans_Omp_CUDA

## Abstract 
This paper is focused on the analysis of the **Kmeans clustering algorithm** and on the study of its performance.
In particular, in addition to the sequential version, we developed other implementations of the algorithm exploiting the parallelism of processors and **GPUs** through the programming languages **Omp** and **CUDA**.

## Results
The Kmeans clustering algorithm's performance has been improved by **multi-processors parallelism** obtained through the **Omp** framework, which allowed to obtain a speedup close to the number of processors avaible. Subsequently, through the parallelism of the **GPUs** and the parallel computing platform **CUDA**, we developed a parallel version of the algorithm achieving significantly higher speedups, in the order of hundreds times, compared to both the sequential and the Omp parallel version. Finally, to achive even higher speedup results, we developed a version of the algorithm making use of the **shared memory** in CUDA and compare it with the previous versions. Pls refer to the document *Parallel_relazione_kmeans.pdf* for more details.

<img align="center" src="https://github.com/divanoLetto/Parallel_Computing_Kmeans_Omp_CUDA/blob/master/Images/presentazione.png" width="90%" height="90%">
