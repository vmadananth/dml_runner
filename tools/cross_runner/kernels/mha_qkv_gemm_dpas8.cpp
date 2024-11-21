

#include <cm/cm.h>
#include <cm/cmtl.h>

#define MATH_E 2.718281828459045235360287471352f
#define FLOAT_MAX 3.402823466e+38f

#define DT half
#define DT_ACCU float
#define SIZE_OF_FP16_BYTE 2
#define EXEC_SIZE 8
#define CONTIGUOUS_K_SIZE 16
#define CONTIGUOUS_V_SIZE  8


// dpas(acc, src1, src2)
// matB is src1 -> (systolicdepth * opsperchannel, exec_size)
// matA is src2 -> (repeatcnt, systolicdepth * opsperchannel) 
_GENX_ inline void myDPAS8(matrix_ref<half, 8, 16> matA,
    matrix_ref<half, 8, 16> matB,
    matrix_ref<float, EXEC_SIZE, 8> result)
{
	
    result = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(result.format<float>(), matB.format<U32>(), matA.format<U32>());

}

extern "C" _GENX_MAIN_ void
mha_qkv_gemm_dpas8(
	uint64_t INMTXa [[type("svmptr_t half")]],  // 0 input qkv surface
	uint64_t OUTMTX [[type("svmptr_t half")]]   // 1 output qxk surface
)
{

	const uint32_t global_x = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
	const uint32_t global_y = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
	const uint32_t global_z = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);

	int linear_global_id = cm_linear_global_id();


	// identifies the block of current thread.
	const uint32_t thread_b = global_z / NUM_HEADS;
	// identfies the head of current thread
	const uint32_t thread_h = global_z % NUM_HEADS;
	// identifies the sequence length of current thread.
	const uint32_t thread_seq = global_y * 8;


	const uint64_t q_base = INMTXa + SIZE_OF_FP16_BYTE * HEAD_SIZE * 3 * (thread_h + NUM_HEADS * MAX_SEQ * thread_b);
	const uint64_t k_base = q_base + SIZE_OF_FP16_BYTE * HEAD_SIZE;
	const uint64_t v_base = k_base + SIZE_OF_FP16_BYTE * HEAD_SIZE;
	const uint64_t out_base = OUTMTX + SIZE_OF_FP16_BYTE * HEAD_SIZE * (thread_h + NUM_HEADS * SEQUENCE_LENGTH * thread_b);


	vector<uint, 8> read_Q_msg;
	read_Q_msg(0) = (HEAD_SIZE * SIZE_OF_FP16_BYTE) - 1; // surface width in bytes - 1
	read_Q_msg(1) = SEQUENCE_LENGTH - 1; // surface height in elements - 1
	read_Q_msg(2) = (HEAD_SIZE * 3 * NUM_HEADS * SIZE_OF_FP16_BYTE) - 1; // surface pitch in bytes - 1
	read_Q_msg(3) = 0; // startX
	read_Q_msg(4) = thread_seq; // startY


	vector<uint, 8> read_K_msg;
	read_K_msg(0) = (HEAD_SIZE * SIZE_OF_FP16_BYTE) - 1; // surface width in bytes - 1
	read_K_msg(1) = KV_SEQUENCE_LENGTH - 1; // surface height in elements - 1
	read_K_msg(2) = (HEAD_SIZE * 3 * NUM_HEADS * SIZE_OF_FP16_BYTE) - 1; // surface pitch in bytes - 1
	read_K_msg(3) = 0; // startX
	read_K_msg(4) = 0; // startY


	vector<uint, 8> read_V_msg;
	read_V_msg(0) = (HEAD_SIZE * SIZE_OF_FP16_BYTE) - 1; // surface width in bytes - 1
	read_V_msg(1) = KV_SEQUENCE_LENGTH - 1; // surface height in elements - 1
	read_V_msg(2) = (HEAD_SIZE * 3 * NUM_HEADS * SIZE_OF_FP16_BYTE) - 1; // surface pitch in bytes - 1
	read_V_msg(3) = 0; // startX
	read_V_msg(4) = 0; // startY

	vector<uint, 8> write_msg;
	write_msg(0) = (HEAD_SIZE * SIZE_OF_FP16_BYTE) - 1; // surface width in bytes - 1
	write_msg(1) = SEQUENCE_LENGTH - 1; // surface height in elements - 1
	write_msg(2) = (HEAD_SIZE * NUM_HEADS * SIZE_OF_FP16_BYTE) - 1; // surface pitch in bytes - 1
	write_msg(3) = 0; // startX
	write_msg(4) = thread_seq; // startY




	// 8 (seqlen) * (head_size)
	matrix<DT, 8, HEAD_SIZE + HEAD_SIZE % CONTIGUOUS_K_SIZE > input_q;
	// cols = 2*EXEC_SIZE
	matrix<DT, 8, 16> input_k1;
	matrix<DT, 8, 16> input_k2;
	matrix<DT, 8, 16> input_v;
	matrix<DT_ACCU, 8, 8> acc_s1;
	matrix<DT_ACCU, 8, 8> acc_s2;
	matrix<DT_ACCU, 8, 16> acc_s;
	input_q = 0;
	input_k1 = 0;
	input_k2 = 0;
	input_v = 0;

	matrix<DT_ACCU, 8, CONTIGUOUS_K_SIZE> p;
	matrix<DT, 8, CONTIGUOUS_K_SIZE> p_half;
	vector<DT_ACCU, 8> m_prev = (0 - FLOAT_MAX);  // m --> max
	vector<DT_ACCU, 8> m_cur;                     // m --> max
	vector<DT_ACCU, 8> f = 0;                     // f --> exp(m_prev - m_cur); 
	vector<DT_ACCU, 8> l_prev = 0;                // l --> sum of exp(Xi-m)
	vector<DT_ACCU, 8> l_cur = 0;                     // l --> sum of exp(Xi-m)
	matrix<DT_ACCU, 8, HEAD_SIZE> acc;
	acc = 0;


	uint start_q_index = thread_seq * (read_Q_msg(2) + 1);
	uint start_out_row = thread_seq * (read_Q_msg(2) + 1);
	uint start_block_k1index = 0;
	uint start_block_k2index = 0;
	uint start_v_index = 0;

	half* q_base_ptr = reinterpret_cast<half*>(q_base);
	for (int i = 0; i < 8; i++) {
		int local_q_index = start_q_index;
		for (int j = 0; j < (HEAD_SIZE + CONTIGUOUS_K_SIZE - 1) / CONTIGUOUS_K_SIZE; ++j)
		{
			input_q.row(i).select<CONTIGUOUS_K_SIZE, 1>(j * CONTIGUOUS_K_SIZE).format<uint32_t>() = cm_ptr_load<uint32_t, CONTIGUOUS_K_SIZE / 2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>((uint32_t*)q_base, local_q_index);

			
			local_q_index += CONTIGUOUS_K_SIZE * SIZE_OF_FP16_BYTE;
		}
		start_q_index += (read_Q_msg(2)) + 1;
	}

#if HEAD_SIZE % CONTIGUOUS_K_SIZE != 0
	input_q.row(0).select< HEAD_SIZE% CONTIGUOUS_K_SIZE, 1>(HEAD_SIZE).format<DT>() = 0;
	input_q.row(1).select< HEAD_SIZE% CONTIGUOUS_K_SIZE, 1>(HEAD_SIZE).format<DT>() = 0;
	input_q.row(2).select< HEAD_SIZE% CONTIGUOUS_K_SIZE, 1>(HEAD_SIZE).format<DT>() = 0;
	input_q.row(3).select< HEAD_SIZE% CONTIGUOUS_K_SIZE, 1>(HEAD_SIZE).format<DT>() = 0;
	input_q.row(4).select< HEAD_SIZE% CONTIGUOUS_K_SIZE, 1>(HEAD_SIZE).format<DT>() = 0;
	input_q.row(5).select< HEAD_SIZE% CONTIGUOUS_K_SIZE, 1>(HEAD_SIZE).format<DT>() = 0;
	input_q.row(6).select< HEAD_SIZE% CONTIGUOUS_K_SIZE, 1>(HEAD_SIZE).format<DT>() = 0;
	input_q.row(7).select< HEAD_SIZE% CONTIGUOUS_K_SIZE, 1>(HEAD_SIZE).format<DT>() = 0;
#endif
	/*
	if (linear_global_id == 0 && global_x == 0 && global_y == 0 && global_z == 0) {
		for (int row = 0; row < 8; row++)
			for (int col = 0; col < 48; col++)
				printf("input_q(%d,%d): %hf \n", row, col, input_q(row, col));
	}*/
	
	//int start_block_k1index = 0;
	//int start_block_k2index = 0;
	half* k_base_ptr = reinterpret_cast<half*>(k_base);
	// should it be 8 or 16
#pragma unroll
	for (int j = 0; j < KV_SEQUENCE_LENGTH; j += CONTIGUOUS_K_SIZE) 

	{
		if (linear_global_id == 0 && global_x == 0 && global_y == 0 && global_z == 0) {

			printf("l_prev at the begining of j: %d \n", j);
			for (int row = 0; row < 8; row++) {
				printf("l_prev: %f \n", l_prev[row]);
				//printf("l_curr: %f \n", l_cur[row]);
			}
		}

		acc_s1 = 0;
		acc_s2 = 0;
		start_block_k1index = j * (read_K_msg(2) + 1);
		start_block_k2index = (j + 8) * (read_K_msg(2) + 1);
#pragma unroll
		for (int block = 0; block < (HEAD_SIZE + CONTIGUOUS_K_SIZE - 1) / CONTIGUOUS_K_SIZE; block++) {
			//for (int block = 0; block < 1; block++) {
			int start_k1row_index = start_block_k1index;
			int start_k2row_index = start_block_k2index;

			for (int i = 0; i < EXEC_SIZE; i++) {
				input_k1.select<CONTIGUOUS_K_SIZE / 2, 1, 2, 1>(0, i * 2).format<U32>() = cm_ptr_load<uint32_t, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>((uint32_t*)k_base, start_k1row_index);
				input_k2.select<CONTIGUOUS_K_SIZE / 2, 1, 2, 1>(0, i * 2).format<U32>() = cm_ptr_load<uint32_t, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>((uint32_t*)k_base, start_k2row_index);

				start_k1row_index += read_K_msg(2) + 1;
				start_k2row_index += read_K_msg(2) + 1;


			}

			myDPAS8(input_q.select<8, 1, 16, 1>(0, block * 16), input_k1, acc_s1);
			myDPAS8(input_q.select<8, 1, 16, 1>(0, block * 16), input_k2, acc_s2);
			start_block_k1index += CONTIGUOUS_K_SIZE * SIZE_OF_FP16_BYTE;
			start_block_k2index += CONTIGUOUS_K_SIZE * SIZE_OF_FP16_BYTE;

		}

		acc_s.select<8, 1, 8, 1>(0, 0) = acc_s1;
		acc_s.select<8, 1, 8, 1>(0, 8) = acc_s2;

		acc_s *= (float)ALPHA;
		cm_vector(mask, ushort, CONTIGUOUS_K_SIZE, 0, 1);
		mask = (j + mask >= KV_SEQUENCE_LENGTH);

		constexpr float float_min = (0 - FLOAT_MAX);
#pragma unroll
		for (int i = 0; i < 8; ++i)
		{
			acc_s.select<1, 1, CONTIGUOUS_K_SIZE, 1>(i, 0).format<DT_ACCU>().merge(float_min, mask);
			m_cur(i) = cm_reduced_max<float>(acc_s.select<1, 1, CONTIGUOUS_K_SIZE, 1>(i, 0).format<DT_ACCU>());
		}

		m_cur.merge(m_prev, m_prev > m_cur);
		f = cm_pow((DT_ACCU)MATH_E, (m_prev - m_cur));
		l_prev *= f;

	

#pragma unroll
		for (int i = 0; i < 8; ++i)
		{
			p.select<1, 1, CONTIGUOUS_K_SIZE, 1>(i, 0).format<DT_ACCU>() = cm_pow((DT_ACCU)MATH_E, (acc_s.select<1, 1, CONTIGUOUS_K_SIZE, 1>(i, 0).format<DT_ACCU>() - m_cur(i)));
			// Masking for p
			p.select<1, 1, CONTIGUOUS_K_SIZE, 1>(i, 0).format<DT_ACCU>().merge(0, mask);
			l_cur(i) = l_prev(i) + cm_sum<float>(p.select<1, 1, CONTIGUOUS_K_SIZE, 1>(i, 0).format<DT_ACCU>());
			acc.select<1, 1, HEAD_SIZE, 1>(i, 0).format<DT_ACCU>() *= f(i);
		}
		p_half = p;


		if (linear_global_id == 0 && global_x == 0 && global_y == 0 && global_z == 0) {

			printf("l_prev and l_cur after change in this iteration of j \n");
			for (int row = 0; row < 8; row++) {
				printf("l_prev: %f \n", l_prev[row]);
				printf("l_curr: %f \n", l_cur[row]);
			}
		}

		cm_fence();

		// Second matmul: P*V
		//int start_v_index = 0;
	    //// Since the input_v should load number of rows = number of exec size.CONTIGUOUS_V_SIZE=8
		// number of blocks for this would be 5
		
		half* v_base_ptr = reinterpret_cast<half*>(v_base);
		start_v_index =  j * (read_V_msg(2) + 1);
		#pragma unroll
		for (int block = 0; block < (HEAD_SIZE + (CONTIGUOUS_V_SIZE) - 1) / (CONTIGUOUS_V_SIZE); block++)
		//for (int block = 0; block < 5; block++)
		{
			int local_v1_index = start_v_index;
			int local_v2_index = local_v1_index + read_V_msg(2)+1;

			for (int i = 0; i < EXEC_SIZE; i++) {
				//input_v.row(i).select<CONTIGUOUS_V_SIZE, 2>(block * CONTIGUOUS_V_SIZE).format<DT>() = cm_ptr_load<U32, CONTIGUOUS_V_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>((uint32_t*)v_base, local_v_index);
				//input_v.row(i).select<CONTIGUOUS_V_SIZE, 2>((block * CONTIGUOUS_V_SIZE)+1).format<DT>() = cm_ptr_load<U32, CONTIGUOUS_V_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>((uint32_t*)v_base, local_v_index + read_V_msg(2)+1);
				input_v.select<1, 1, CONTIGUOUS_V_SIZE * 2, 2>(i, 0).format<U32>() = cm_ptr_load<uint32_t, CONTIGUOUS_V_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>((uint32_t*)v_base, local_v1_index);
				input_v.select<1, 1, CONTIGUOUS_V_SIZE * 2, 2>(i, 1).format<U32>() = cm_ptr_load<uint32_t, CONTIGUOUS_V_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>((uint32_t*)v_base, local_v2_index);

				local_v1_index += (read_V_msg(2)+1) * 2;
				local_v2_index += (read_V_msg(2)+1) * 2;
			}
	
			myDPAS8(p_half, input_v, acc.select<8, 1, CONTIGUOUS_V_SIZE, 1>(0, block * CONTIGUOUS_V_SIZE));
			start_v_index += CONTIGUOUS_V_SIZE * SIZE_OF_FP16_BYTE;
		}
		// unexpected behavior of l_cur , every other itrationo of j resets l_cur after DPAS 
		if (linear_global_id == 0 && global_x == 0 && global_y == 0 && global_z == 0) {

			printf("l_prev before exchange %d \n", j );	
			for (int row = 0; row < 8; row++) {
				printf("l_prev: %f \n", l_prev[row]);
				printf("l_curr: %f \n", l_cur[row]);
			}
		}
		
		m_prev = m_cur;
		l_prev = l_cur;

	}

	/*if (linear_global_id == 0 && global_x == 0 && global_y == 0 && global_z == 0) {
		for (int row = 0; row < 8; row++) {
			for (int col = 0; col < 40; col++) {
				printf("acc_out(%d,%d): %hf \n", row, col, acc(row, col));
			}
		}
	}

	if (linear_global_id == 0 && global_x == 0 && global_y == 0 && global_z == 0) {
		for (int row = 0; row < 8; row++) {
			printf("l_prev: %f \n", l_prev[row]);
			//printf("l_curr: %f \n", l_cur[row]);
		}
	}*/

	matrix<DT_ACCU, 8, HEAD_SIZE> acc_out = acc;
#pragma unroll
	for (int j = 0; j < 8; ++j)
	{
		acc_out.select<1, 1, HEAD_SIZE, 1>(j, 0) /= l_prev(j);
	}

#pragma unroll
	for (int block = 0; block < (HEAD_SIZE + CONTIGUOUS_K_SIZE - 1) / CONTIGUOUS_K_SIZE; ++block)
	{
		matrix<DT, 8, CONTIGUOUS_K_SIZE> out_tile = 0;
		out_tile = acc_out.select<8, 1, CONTIGUOUS_K_SIZE, 1>(0, block * CONTIGUOUS_K_SIZE);
		/*if (linear_global_id == 0 && global_x == 0 && global_y == 0 && global_z == 0) {
			for (int row = 0; row < 8; row++) {
				for (int col = 0; col < 16; col++) {
					printf("out_tile(%d,%d): %hf \n", row, col, out_tile(row, col));
				}
			}
		}*/
		//cm_store<DT, CONTIGUOUS_K_SIZE, 8, CacheHint::WriteBack, CacheHint::WriteBack>((DT*)out_base, write_msg(0), write_msg(1), write_msg(2), write_msg(3), write_msg(4), out_tile.format<DT>());
		int local_out_index = start_out_row;
		for (int i = 0; i < 8; i++)
		{
			cm_ptr_store<U32, CONTIGUOUS_K_SIZE/2, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>((uint32_t*)out_base, 0, out_tile.select<1, 1, CONTIGUOUS_K_SIZE, 1>(i, block* CONTIGUOUS_K_SIZE).format<U32>());
			//cm_ptr_store<U32, CONTIGUOUS_K_SIZE, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>((uint32_t*)out_base, local_out_index, out_tile.select<1,1,CONTIGUOUS_K_SIZE, 1>(i, block * CONTIGUOUS_K_SIZE).format<DT>());
			local_out_index = local_out_index + (write_msg(2)+1);
		}

		start_out_row += CONTIGUOUS_K_SIZE * SIZE_OF_FP16_BYTE;
	}	
	
}
	