
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <unordered_map>
#include <bitset>
#include <thread>
#include <math.h> 

#define BITS 64
#define MAX_LEN_WORD 20

using namespace std;

string readFile(string file) {
    ifstream MyFile;
    MyFile.open(file);
    stringstream strStream;
    strStream << MyFile.rdbuf();
    return strStream.str();
}

vector<string> strToVector(string data, char stop_w = '\n') {
    vector<string> out;
    size_t last_init = 0;
    for (size_t i = 0; i < data.length(); i++) {
        if (data[i] == stop_w) {
            string temp = data.substr(last_init, i - last_init);
            out.push_back(temp);
            last_init = i + 1;
        }
    }
    return out;
}

void printVector(vector<string> data) {
    for (size_t i = 0; i < data.size(); i++) {
        printf("%s\n", data[i].c_str());
    }
}

void printVectorInt(vector<int> data) {
    for (size_t i = 0; i < data.size(); i++) {
        printf("%i ", data[i]);
    }
    printf("\n");
}

void printMap(unordered_map <string, int> data) {
    for (unordered_map<string, int >::const_iterator it = data.begin(); it != data.end(); ++it) {
        cout << it->first << " " << it->second << "\n";
    }
}

string* text_generator(vector<string> my_dict, int len_words) {
    string* text = new string[len_words];
    for (size_t i = 0; i < len_words; i++) {
        text[i] = my_dict[rand() % my_dict.size()];
    }
    return text;
}

unordered_map <string, int> get_frec(string* words, int len_words) {

    unordered_map <string, int> out;
    for (size_t i = 0; i < len_words; i++) {
        if (out.count(words[i]) > 0) { // cuantas veces parece la palabra en el texto
            out[words[i]] += 1;
        }
        else {
            out[words[i]] = 1;
        }
    }
    return out;
}

string sum_matrix(vector<long*> in_matrix) {
    string out;
    for (size_t j = 0; j < BITS; j++) {
        long int temp = 0;
        for (size_t i = 0; i < in_matrix.size(); i++) {
            temp += in_matrix[i][j];
        }
        if (temp > 0) {
            out += '1';
        }
        else {
            out += '0';
        }
    }
    return out;
}

size_t count_words(unordered_map <string, int>* in_words, int n_text) {
    size_t count_l = 0;
    for (size_t i = 0; i < n_text; i++) {
        count_l += in_words[i].size();
    }
    return count_l;
}

char* strToChar(string data) {
    char* out = new char[MAX_LEN_WORD];
    for (size_t i = 0; i < MAX_LEN_WORD; i++) {
        if (i < data.size()) {
            out[i] = data[i];
        }
        else {
            out[i] = 0;
        }
    }
    return out;
}

void compress_sim_data_cuda(unordered_map <string, int>* in_words, char * & s_out, int * & f_out , int n_text) {

    size_t numerate = 0;
    for (size_t i = 0; i < n_text; i++) {
        for (unordered_map<string, int >::const_iterator it = in_words[i].begin(); it != in_words[i].end(); ++it) {
            char* t_str = strToChar(it->first);
            for (size_t i = 0; i < MAX_LEN_WORD; i++) {
                s_out[numerate * MAX_LEN_WORD + i] = t_str[i];
            }
            f_out[numerate] = it->second;
            numerate++;
        }
    }
}

string* extract_sim_data_cuda(long * v_words, unordered_map <string, int>* in_words, int n_text) {
    string* out = new string[n_text];
    size_t numerate = 0;
    for (size_t i = 0; i < n_text; i++) {
        vector<long*> temp;
        for (unordered_map<string, int >::const_iterator it = in_words[i].begin(); it != in_words[i].end(); ++it) {
            long * t_bits = new long[BITS];
            for (size_t b = 0; b < BITS; b++) {
                t_bits[b] = v_words[numerate + b];
            }
            temp.push_back(t_bits);
            numerate+= BITS;
        }
        out[i] = sum_matrix(temp);
    }
    return out;
}


void sim_hash_lineal(char* s_in, int* f_in, long* & out, int len) {
    for (size_t t = 0; t < len; t++) {
        unsigned long long int hash = 5381;
        for (size_t i = 0; i < MAX_LEN_WORD; i++) {
            if (s_in[t * MAX_LEN_WORD + i] != 0) {
                hash = ((hash << 5) + hash) + (int)s_in[t * MAX_LEN_WORD + i];
            }
        }
        bool* bits = new bool[BITS];
        for (size_t i = 0; i < BITS; i++) {
            bits[i] = hash % 2;
            hash = hash / 2;
        }
        for (size_t i = 0; i < BITS; i++) {
            size_t p = (t * BITS) + i;
            out[p] = (int)bits[i];
            if (out[p] == 1) {
                out[p] += f_in[t];
            }
            else {
                out[p] -= f_in[t];
            }
        }
    }
}

__global__ void cuda_sim_hash(char * s_in, int * f_in, long * out, int len)
{
    int t = (blockIdx.x * blockDim.x) + (threadIdx.x);
    if (t >= 0 && t < len) {
        unsigned long long int hash = 5381;
        for (size_t i = 0; i < MAX_LEN_WORD; i++) {
            if (s_in[t * MAX_LEN_WORD + i] != 0) {
                hash = ((hash << 5) + hash) + (int)s_in[t * MAX_LEN_WORD + i];
            }
        }
        bool* bits = new bool[BITS];
        for (size_t i = 0; i < BITS; i++) {
            bits[i] = hash % 2;
            hash = hash / 2;
        }
        for (size_t i = 0; i < BITS; i++) {
            size_t p = (t * BITS) + i;
            out[p] = (int)bits[i];
            if (out[p] == 1) {
                out[p] += f_in[t];
            }
            else {
                out[p] -= f_in[t];
            }
        }
        delete bits;
    }
}


bool compare_str(string * a, string * b, int len) {
    for (size_t i = 0; i < len; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

int main() {

    clock_t begin, end;
    double elapsed_secs;
    long long int w_s; //palabras por segundo

    srand(time(NULL));
    
    string words = readFile("words.txt"); //leer palabras
    vector<string> l_words = strToVector(words); //libreria de palabras

    int long_text = 4000; //longitud de palabras por texto
    int n_text = 2560; //cantidad de textos (documentos)

    printf("Num. textos: %i, Long Text: %i \n", n_text, long_text);

    unordered_map <string, int>* words_frec = new unordered_map <string, int>[n_text]; 
    // diccionario de frecuencias por palabra de cada documento

    for (size_t i = 0; i < n_text; i++) {
        words_frec[i] = get_frec(text_generator(l_words, long_text), long_text);
    }

    size_t amount_words = count_words(words_frec, n_text);

    printf("Total Words: %i words \n", amount_words);

    char * s_in = new char[amount_words * MAX_LEN_WORD];
    int * f_in = new int[amount_words];
    long * out = new long[amount_words * BITS];

    compress_sim_data_cuda(words_frec, s_in, f_in , n_text);

    char* cu_s_in = 0;
    int* cu_f_in = 0;
    long* cu_out = 0;

    cudaMalloc((void**)&cu_s_in, amount_words * sizeof(char) * MAX_LEN_WORD);
    cudaMalloc((void**)&cu_f_in, amount_words * sizeof(int));
    cudaMalloc((void**)&cu_out, amount_words * sizeof(long) * BITS);

    begin = clock();

    cudaMemcpy(cu_s_in, s_in, amount_words * sizeof(char) * MAX_LEN_WORD, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_f_in, f_in, amount_words * sizeof(int), cudaMemcpyHostToDevice);

    int thr = 1024;
    int dim_grid = (amount_words/thr)+1;    

    cuda_sim_hash <<< dim_grid, thr >>> (cu_s_in, cu_f_in, cu_out, amount_words);

    cudaMemcpy(out, cu_out, amount_words * sizeof(long) * BITS, cudaMemcpyDeviceToHost);

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("Grilla: %d, Bloque:  %d \n", dim_grid, thr);
    printf("Tiempo Cuda:  %f ms \n", elapsed_secs);
    w_s = amount_words / elapsed_secs;
    printf("palabras por Seg:  %d words \n", w_s);
    
    string* r_out = extract_sim_data_cuda(out, words_frec, n_text);



    long* out_l = new long[amount_words * BITS];

    begin = clock();

    sim_hash_lineal(s_in, f_in, out_l, amount_words);

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("Tiempo Lineal:  %f ms \n", elapsed_secs);
    w_s = amount_words / elapsed_secs;
    printf("palabras por Seg:  %d words \n", w_s);

    string* r_out_l = extract_sim_data_cuda(out_l, words_frec, n_text);

    /*
    for (size_t i = 0; i < n_text; i++) {
        cout << r_out_l[i] << endl;
    }*/

    if (compare_str(r_out, r_out_l, n_text)) {
        cout << "Ok" << endl;
    }
    else {
        cout << "Error" << endl;
    }
}

