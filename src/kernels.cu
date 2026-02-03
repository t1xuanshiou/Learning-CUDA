#include <vector>
#include <cuda_fp16.h>
#include "../tester/utils.h"

#define BLOCK_SIZE 32

template <typename T>
inline __device__ T typed_max(T a, T b) {
  return max(a, b);
}

template <typename T>
inline __device__ T typed_exp(T x) {
  return exp(x);
}

template <>
inline __half __device__ typed_max<__half>(__half a, __half b) {
  return __hgt(a, b) ? a : b;
}

template <>
inline __half __device__ typed_exp<__half>(__half x) {
  return __float2half(exp(__half2float(x)));
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */

 template <typename T>
__device__ T warp_reduce(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__global__ void traceKernel(T* d_result, const T* d_input, size_t rows, size_t cols, size_t n) {
    extern __shared__ char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    T val = (idx < n) ? d_input[idx * cols + idx] : (T)0;
    T warp_sum = warp_reduce(val);
    if (tid % 32 == 0) {
        smem[tid / 32] = warp_sum;
    }
    __syncthreads();

    if (tid < 32) {
        int num_warps = (blockDim.x + 31) / 32;
        T block_sum = (tid < num_warps) ? smem[tid] : (T)0;
        block_sum = warp_reduce(block_sum);
        if (tid == 0) {
            atomicAdd(d_result, block_sum);
        }
    }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    size_t n = std::min(rows, cols);
    if (n == 0) return (T)0;

    cudaStream_t stream;
    RUNTIME_CHECK(cudaStreamCreate(&stream));
    
    T *d_input, *d_result;
    size_t matrixSize = rows * cols * sizeof(T);

    RUNTIME_CHECK(cudaMallocAsync(&d_input, matrixSize, stream));
    RUNTIME_CHECK(cudaMallocAsync(&d_result, sizeof(T), stream));
    RUNTIME_CHECK(cudaMemcpyAsync(d_input, h_input.data(), matrixSize, cudaMemcpyHostToDevice, stream));
    RUNTIME_CHECK(cudaMemsetAsync(d_result, 0, sizeof(T), stream));
    int blockSize = 1024;
    if (n < blockSize) {
        blockSize = ((n + 31) / 32) * 32;
    }

    int gridSize = (n + blockSize - 1) / blockSize;
    
    int numWarpsPerBlock = (blockSize + 31) / 32;
    int sharedMemSize = numWarpsPerBlock * sizeof(T);
    
    traceKernel<T><<<gridSize, blockSize, sharedMemSize, stream>>>(
        d_result, d_input, rows, cols, n
    );
    
    T h_result = 0;
    RUNTIME_CHECK(cudaMemcpyAsync(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost, stream));
    RUNTIME_CHECK(cudaStreamSynchronize(stream));
    RUNTIME_CHECK(cudaFreeAsync(d_input, stream));
    RUNTIME_CHECK(cudaFreeAsync(d_result, stream));
    RUNTIME_CHECK(cudaStreamDestroy(stream));
    
    return h_result;
}



/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

//blockThread为(32,32)

//采用falsh attention v2的思路实现
//见遍历Q,再K,V
template <typename T>
__global__ void flashAttentionKernel(T* d_q, T* d_k, T* d_v, T* d_o, 
                                     int batch_size, int target_seq_len, int src_seq_len, 
                                     int query_heads, int kv_heads, int head_dim, 
                                     bool is_causal, int ratio, float attention_scale) { 

  int current_batch = blockIdx.x;
  int current_q_head = blockIdx.y;
  int current_kv_head = current_q_head / ratio;

  extern __shared__ char smem_raw[];
  
  T* smem_q = reinterpret_cast<T*>(smem_raw);
  T* smem_k = smem_q + BLOCK_SIZE * head_dim;
  T* smem_v = smem_k + BLOCK_SIZE * head_dim;
  
  char* curr_ptr = (char*)(smem_v + BLOCK_SIZE * head_dim);
  
  float* smem_s = reinterpret_cast<float*>(curr_ptr); 
  float* smem_m = smem_s + BLOCK_SIZE * BLOCK_SIZE;
  float* smem_l = smem_m + BLOCK_SIZE;
  float* smem_o = smem_l + BLOCK_SIZE;

  int base_q_offset = current_batch * target_seq_len * query_heads * head_dim + current_q_head * head_dim;
  int base_kv_offset = current_batch * src_seq_len * kv_heads * head_dim + current_kv_head * head_dim;

  for(int i = 0; i < target_seq_len; i += BLOCK_SIZE){
    
    // Load Q
    // 由于是一个block一个block处理Q的，要考虑越界
    bool is_valid_q_row = (i + threadIdx.y < target_seq_len);
    for(int k = 0; k < head_dim; k += BLOCK_SIZE){
      if (threadIdx.x + k < head_dim) {
          int cur_q_idx = base_q_offset + (i + threadIdx.y) * query_heads * head_dim + k + threadIdx.x;
          smem_q[threadIdx.y * head_dim + threadIdx.x + k] = 
              (is_valid_q_row && cur_q_idx < batch_size * target_seq_len * query_heads * head_dim) ? d_q[cur_q_idx] : (T)0;
      }
    }

    //由第0号线程初始化m,l,o
    if(threadIdx.x == 0){
      smem_m[threadIdx.y] = -INFINITY;
      smem_l[threadIdx.y] = 0.0f;
    }
    for(int k = 0; k < head_dim; k += BLOCK_SIZE){
      if (threadIdx.x + k < head_dim)
          smem_o[threadIdx.y * head_dim + threadIdx.x + k] = 0.0f;
    }
    __syncthreads();

    int j_start = 0;
    int j_end = src_seq_len;
    
    if (is_causal) {
      int max_valid_col = i + BLOCK_SIZE - 1;
      
      j_end = ((max_valid_col / BLOCK_SIZE) + 1) * BLOCK_SIZE;
      j_end = min(j_end, src_seq_len);
    }

    for(int j = j_start; j < j_end; j += BLOCK_SIZE){
      //跳过被mask的部分，不参加计算
      if (is_causal) {

        int min_q_row = i;
        int min_kv_col = j;
        
        if (min_q_row < min_kv_col) {
          continue;
        }
      }
      // Load K,V, K,V都是(block_size, head_dim)
      bool is_valid_kv_row = (j + threadIdx.y < src_seq_len);
      for(int k = 0; k < head_dim && k + threadIdx.x < head_dim; k += BLOCK_SIZE){
        int cur_kv_idx = base_kv_offset + (j + threadIdx.y) * kv_heads * head_dim + k + threadIdx.x;
        smem_k[threadIdx.y * head_dim + threadIdx.x + k] = 
            (is_valid_kv_row && cur_kv_idx < batch_size * src_seq_len * kv_heads * head_dim) ? d_k[cur_kv_idx] : (T)0;
        smem_v[threadIdx.y * head_dim + threadIdx.x + k] = 
            (is_valid_kv_row && cur_kv_idx < batch_size * src_seq_len * kv_heads * head_dim) ? d_v[cur_kv_idx] : (T)0;
      }
      __syncthreads();

      // Compute S (float)
      float s_ij = 0.0f;
      for(int k = 0; k < head_dim; k += BLOCK_SIZE){
        for(int l = 0; l < BLOCK_SIZE && k + l < head_dim; l++){
          s_ij += (float)smem_q[threadIdx.y * head_dim + k + l] * (float)smem_k[threadIdx.x * head_dim + k + l];
        }
      }
      s_ij *= attention_scale;

      // Masking
      int global_row = i + threadIdx.y;
      int global_col = j + threadIdx.x;
      
      // 超出src_seq_len的部分mask掉，这里不能为0!!!!!!!!,要设置为-Infinity，不然exp(0)=1
      if (global_col >= src_seq_len) s_ij = -INFINITY;
      
      if (is_causal) {
        int max_q_row = i + BLOCK_SIZE - 1;
        int min_kv_col = j;
        
        if (max_q_row >= min_kv_col) {
          if (global_row < global_col) {
            s_ij = -INFINITY;
          }
          // s_ij = -INFINITY;
        }
        
      }

      // 得到block_size行的max和sum，更新m,l
      float row_max = s_ij;
      for(int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1){
        row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
      }
      row_max = __shfl_sync(0xffffffff, row_max, 0);

      float m_old = smem_m[threadIdx.y];
      float m_new = fmaxf(m_old, row_max);

      s_ij = expf(s_ij - m_new);

      float row_sum = s_ij;
      for(int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1){
        row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
      }
      row_sum = __shfl_sync(0xffffffff, row_sum, 0);

      float l_old = smem_l[threadIdx.y];
      float rescale_factor = expf(m_old - m_new);
      float l_new = row_sum + rescale_factor * l_old;

      smem_s[threadIdx.y * BLOCK_SIZE + threadIdx.x] = s_ij; 
      //由第0号线程更新m,l
      if (threadIdx.x == 0) {
          smem_m[threadIdx.y] = m_new; 
          smem_l[threadIdx.y] = l_new; 
      }
      __syncthreads();

      // Update O
      for(int k = 0; k < head_dim && k + threadIdx.x < head_dim; k += BLOCK_SIZE){
        float o_ij = smem_o[threadIdx.y * head_dim + k + threadIdx.x] * rescale_factor;
        
        for(int l = 0; l < BLOCK_SIZE; l++){
          o_ij += smem_s[threadIdx.y * BLOCK_SIZE + l] * float(smem_v[l * head_dim + k + threadIdx.x]);
        }
        smem_o[threadIdx.y * head_dim + k + threadIdx.x] = o_ij;
      }
      __syncthreads();
    }

    if (is_valid_q_row) {
      for(int k = 0; k < head_dim && threadIdx.x + k < head_dim; k += BLOCK_SIZE){
          float l_val = smem_l[threadIdx.y];
          float final_val = smem_o[threadIdx.y * head_dim + k + threadIdx.x];
          
          if (l_val != 0.0f) {
               final_val /= l_val;
          }
          
          int cur_o_idx = base_q_offset + (i + threadIdx.y) * query_heads * head_dim + k + threadIdx.x;
          if(cur_o_idx < batch_size * target_seq_len * query_heads * head_dim){
              d_o[cur_o_idx] = (T)final_val;
          }
      }
    }
    __syncthreads();
  }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  
  cudaStream_t stream;
  RUNTIME_CHECK(cudaStreamCreate(&stream));
  
  size_t q_size = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
  size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
  size_t o_size = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);

  // 异步内存分配和拷贝
  T* d_q; T* d_k; T* d_v; T* d_o;
  RUNTIME_CHECK(cudaMallocAsync(&d_q, q_size, stream));
  RUNTIME_CHECK(cudaMallocAsync(&d_k, kv_size, stream));
  RUNTIME_CHECK(cudaMallocAsync(&d_v, kv_size, stream));
  RUNTIME_CHECK(cudaMallocAsync(&d_o, o_size, stream));

  RUNTIME_CHECK(cudaMemcpyAsync(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice, stream));
  RUNTIME_CHECK(cudaMemcpyAsync(d_k, h_k.data(), kv_size, cudaMemcpyHostToDevice, stream));
  RUNTIME_CHECK(cudaMemcpyAsync(d_v, h_v.data(), kv_size, cudaMemcpyHostToDevice, stream));
  
  RUNTIME_CHECK(cudaStreamSynchronize(stream));

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(batch_size, query_heads);
  
  int ratio = query_heads / kv_heads; 
  float attention_scale = 1.0f / sqrt((float)head_dim);
  
  size_t smem_bytes_t = (3 * BLOCK_SIZE * head_dim) * sizeof(T);
  size_t smem_bytes_float = (BLOCK_SIZE * BLOCK_SIZE + 2 * BLOCK_SIZE + BLOCK_SIZE * head_dim) * sizeof(float);
  size_t smem_size = smem_bytes_t + smem_bytes_float;
  
  flashAttentionKernel<T><<<grid, block, smem_size, stream>>>(d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal, ratio, attention_scale);
  RUNTIME_CHECK(cudaStreamSynchronize(stream));
  // 异步拷贝结果回主机
  RUNTIME_CHECK(cudaMemcpyAsync(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost, stream));
  
  RUNTIME_CHECK(cudaStreamSynchronize(stream));

  // 异步释放内存和清理资源
  RUNTIME_CHECK(cudaFreeAsync(d_q, stream));
  RUNTIME_CHECK(cudaFreeAsync(d_k, stream));
  RUNTIME_CHECK(cudaFreeAsync(d_v, stream));
  RUNTIME_CHECK(cudaFreeAsync(d_o, stream));
  RUNTIME_CHECK(cudaStreamSynchronize(stream));
  RUNTIME_CHECK(cudaStreamDestroy(stream));
}
// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
