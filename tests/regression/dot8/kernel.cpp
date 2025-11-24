#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	// reinterpret the buffers with explicit types
    auto A = reinterpret_cast<int8_t*>(arg->A_addr);
    auto B = reinterpret_cast<int8_t*>(arg->B_addr);
    auto C = reinterpret_cast<int32_t*>(arg->C_addr);
    const int size = arg->size;

    int col = blockIdx.x;
    int row = blockIdx.y;

    int32_t sum = 0;
    int k = 0;
    // std::cout << " Here ";
    // process blocks of 4 using DOT8
    for (; k + 3 < size; k += 4) {
        // Pack 4 int8_t from A[row, k..k+3] into a 32-bit word
        // We pack bytes in increasing significance: b0 at bits [7:0], b1 at [15:8], etc.
        // Use uint32_t and cast to uint8_t first to avoid sign-extension during shifts.
        uint32_t packedA =
            (uint32_t)(uint8_t)A[row * size + (k + 0)] << 0  |
            (uint32_t)(uint8_t)A[row * size + (k + 1)] << 8  |
            (uint32_t)(uint8_t)A[row * size + (k + 2)] << 16 |
            (uint32_t)(uint8_t)A[row * size + (k + 3)] << 24;

        // Pack 4 int8_t from B[k..k+3, col] (column-major access) similarly
        uint32_t packedB =
            (uint32_t)(uint8_t)B[(k + 0) * size + col] << 0  |
            (uint32_t)(uint8_t)B[(k + 1) * size + col] << 8  |
            (uint32_t)(uint8_t)B[(k + 2) * size + col] << 16 |
            (uint32_t)(uint8_t)B[(k + 3) * size + col] << 24;

        // std::cout << packedA << packedB;
        // Call custom instruction (vx_dot8 expects int arguments)
        int dp = vx_dot8((int)packedA, (int)packedB);
        /*
        int dp = 0;
        for (int i = 0; i < 4; ++i) {
            int8_t a_byte = static_cast<int8_t>((packedA >> (8 * i)) & 0xFF);
            int8_t b_byte = static_cast<int8_t>((packedB >> (8 * i)) & 0xFF);
            dp += a_byte * b_byte;
          }
        */
        // Accumulate into 32-bit sum
        sum += dp;
    }

    /*
    // tail: handle remaining elements (if size not divisible by 4)
    for (; k < size; ++k) {
        // sign-extend each int8_t to int32_t before multiply
        int32_t av = (int32_t)A[row * size + k];
        int32_t bv = (int32_t)B[k * size + col];
        sum += av * bv;
    }
    */
    C[row * size + col] = sum;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
