# Parallel_Computing_Kmeans_Omp_CUDA

## Abstract 
This paper is focused on the analysis of the **Kmeans clustering algorithm** and on the study of its performance.
In particular, in addition to the sequential version, we develope other implementations of the algorithm that exploit the parallelism of processors and **GPUs** through the programming languages **Omp** and **CUDA**.

## Results
The Kmeans clustering algorithm's performance has been improved by **multi-processors parallelism** obteined through the **Omp** framework, which allowed to obtain a speedup close to the number of processors avaible. Subsequently, through the parallelism of the **GPUs**, we develope a version of the algorithm by the parallel computing platform **CUDA**, that allowed us to obtain enormously higher speedups, in the order of hundreds, compared to both the sequential and the Omp parallel version. Finally to achive even higher speedup results we develope a version of the algorithm that use the **shared memory** in CUDA and compare it with the previous versions.   
Read the document *Parallel_relazione_kmeans.pdf* for more details.

<img align="center" src="https://github.com/divanoLetto/Parallel_Computing_Kmeans_Omp_CUDA/blob/master/Images/presentazione.png" width="90%" height="90%">
