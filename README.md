# Sample Codes using NVSHMEM on Multi-GPU

NVSHMEM is a parallel programming interface based on OpenSHMEM that provides efficient and scalable communication for NVIDIA GPU clusters. NVSHMEM creates a global address space for data that spans the memory of multiple GPUs and can be accessed with fine-grained GPU-initiated operations, CPU-initiated operations, and operations on CUDA® streams.

----
## What is NVSHMEM?
see [NVIDIA]( https://developer.nvidia.com/nvshmem)

[NVSHMEM](https://developer.nvidia.com/nvshmem) is a parallel programming model for efficient and scalable communication across multiple NVIDIA GPUs. NVSHMEM, which is based on [OpenSHMEM](http://openshmem.org/site/), provides a global address space for data that spans the memory of multiple GPUs and can be accessed with fine-grained GPU-initiated operations, CPU-initiated operations, and operations on CUDA streams. NVSHMEM offers a compelling multi-GPU programming model for many application use cases, and is especially valuable on modern GPU servers that have a high density of GPUs per server node and complex interconnects such as [NVIDIA NVSwitch](https://www.nvidia.com/en-us/data-center/nvlink/) on the [NVIDIA DGX A100 server](https://www.nvidia.com/en-us/data-center/dgx-a100/).

----

## NVSHMEM on OGBON

----
### Enviroment Variable

> ~$ echo $NVSHMEM_HOME/
> ~$ /opt/share/ucx/1.12.0-cuda-11.6-ofed-5.4/

----
### How to Inicialize Modules

> ~$ module load nvshmem/2.8.0 

> ~$ module list
> ~$ Currently Loaded Modulefiles:
  1) gcc/11.1.0                    3) ucx/1.13.1-cuda-12.0          5) nvshmem/2.8.0
  2) cuda/12.0                     4) openmpi/4.1.4-gcc-cuda-12.0

----
### Hello World in NVSHMEM (helloWorld_nvshmem.cu)

~~~c++
#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

__global__ void simple_shift(int *destination) 
{
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;

    nvshmem_int_p(destination, mype, peer);
}

int main(int argc, char **argv) 
{
    int mype_node, msg;
    cudaStream_t stream;

    nvshmem_init();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);

    int *destination = (int *) nvshmem_malloc(sizeof(int));

    simple_shift<<<1, 1, 0, stream>>>(destination);
    nvshmemx_barrier_all_on_stream(stream);
    cudaMemcpyAsync(&msg, destination, sizeof(int), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    printf("%d: received message %d\n", nvshmem_my_pe(), msg);

    nvshmem_free(destination);
    nvshmem_finalize();
    return 0;
}
~~~

----
### How to Compile

> ~$ nvcc -rdc=true -ccbin g++ -gencode=arch=compute_70,code=sm_70 -I $NVSHMEM_HOME/include -L $NVSHMEM_HOME/lib  helloWorld_nvshmem.cu -o helloWorld_nvshmem -lnvidia-ml -lcudart -lnvshmem -lcuda

----
### How to Execute

> ~$ nvshmrun -n 4 ./helloWorld_nvshmem


## Requirements

NCCL requires at least CUDA 12.0 and Kepler or newer GPUs. For InfiniBand GPUDirect Async (IBGDA) based platforms, best performance is achieved when all GPUs are located on multi-socket configurations.
