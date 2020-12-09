
void kernel opencl_sim_hash(global const char* s_in, global const int* f_in, global long* out, global const int * len){    
    
    int t = get_global_id(0) ;
    if (t < *len) {
        ulong hash = 5381;
        for (unsigned long i = 0; i < 20; i++) {
            if (s_in[t * 20 + i] != 0) {
                hash = ((hash << 5) + hash) + (int)s_in[t * 20 + i];
            }
        }
        //printf("%i %lld\n", t, hash);

        bool bits[64];
        for (int i = 0; i < 64; i++) {
            bits[i] = hash % 2;
            hash = hash / 2;
        }
        for (int i = 0; i < 64; i++) {
            unsigned long p = (t * 64) + i;
            out[p] = (int)bits[i];
            if (out[p] == 1) {
                out[p] += f_in[t];
            }
            else {
                out[p] -= f_in[t];
            }
        }
    }
};
