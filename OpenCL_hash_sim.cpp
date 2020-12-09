#include <CL/cl.hpp>

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
using namespace cl;

string readFile(string file) {
    ifstream MyFile;
    MyFile.open(file);
    stringstream strStream;
    strStream << MyFile.rdbuf();
    return strStream.str();
}

vector<string> strToVector(string data, char stop_w = '\n') {
    vector<string> out;
    std::size_t last_init = 0;
    for (std::size_t i = 0; i < data.length(); i++) {
        if (data[i] == stop_w) {
            string temp = data.substr(last_init, i - last_init);
            out.push_back(temp);
            last_init = i + 1;
        }
    }
    return out;
}

void printVector(vector<string> data) {
    for (std::size_t i = 0; i < data.size(); i++) {
        printf("%s\n", data[i].c_str());
    }
}

void printVectorInt(vector<int> data) {
    for (std::size_t i = 0; i < data.size(); i++) {
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
    for (std::size_t i = 0; i < len_words; i++) {
        text[i] = my_dict[rand() % my_dict.size()];
    }
    return text;
}

unordered_map <string, int> get_frec(string* words, int len_words) {

    unordered_map <string, int> out;
    for (std::size_t i = 0; i < len_words; i++) {
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
    for (std::size_t j = 0; j < BITS; j++) {
        long int temp = 0;
        for (std::size_t i = 0; i < in_matrix.size(); i++) {
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

std::size_t count_words(unordered_map <string, int>* in_words, int n_text) {
    std::size_t count_l = 0;
    for (std::size_t i = 0; i < n_text; i++) {
        count_l += in_words[i].size();
    }
    return count_l;
}

char* strToChar(string data) {
    char* out = new char[MAX_LEN_WORD];
    for (std::size_t i = 0; i < MAX_LEN_WORD; i++) {
        if (i < data.size()) {
            out[i] = data[i];
        }
        else {
            out[i] = 0;
        }
    }
    return out;
}

void compress_sim_data_cuda(unordered_map <string, int>* in_words, char*& s_out, int*& f_out, int n_text, int init = 0) {
    std::size_t numerate = 0;
    for (std::size_t i = init; i < n_text; i++) {
        for (unordered_map<string, int >::const_iterator it = in_words[i].begin(); it != in_words[i].end(); ++it) {
            char* t_str = strToChar(it->first);
            for (std::size_t i = 0; i < MAX_LEN_WORD; i++) {
                s_out[numerate * MAX_LEN_WORD + i] = t_str[i];
            }
            f_out[numerate] = it->second;
            numerate++;
        }
    }
}

string* extract_sim_data_cuda(long* v_words, unordered_map <string, int>* in_words, int n_text, int init = 0) {
    string* out = new string[n_text-init];
    std::size_t numerate = 0;
    for (std::size_t i = init; i < n_text; i++) {
        vector<long*> temp;
        for (unordered_map<string, int >::const_iterator it = in_words[i].begin(); it != in_words[i].end(); ++it) {
            long* t_bits = new long[BITS];
            for (std::size_t b = 0; b < BITS; b++) {
                t_bits[b] = v_words[numerate + b];
            }
            temp.push_back(t_bits);
            numerate += BITS;
        }
        out[i] = sum_matrix(temp);
    }
    return out;
}


void sim_hash_lineal(char* s_in, int* f_in, long*& out, int len) {
    for (std::size_t t = 0; t < len; t++) {
        unsigned long long hash = 5381;
        for (unsigned long i = 0; i < MAX_LEN_WORD; i++) {
            if (s_in[t * MAX_LEN_WORD + i] != 0) {
                hash = ((hash << 5) + hash) + (int)s_in[t * MAX_LEN_WORD + i];
            }
        }
        bool bits[BITS];
        for (int i = 0; i < BITS; i++) {
            bits[i] = hash % 2;
            hash = hash / 2;
        }
        for (int i = 0; i < BITS; i++) {
            unsigned long p = (t * BITS) + i;
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


bool compare_str(string* a, string* b, int len) {
    for (std::size_t i = 0; i < len; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

string kernel_code = readFile("kernel.cl");

int main() {
    //get all platforms (drivers)
    vector<Platform> all_platforms;
    Platform::get(&all_platforms); // leer plataformas
    if (all_platforms.size() == 0) {
        cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    Platform default_platform = all_platforms[0];
    cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    //get default device of the default platform
    vector<Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    Device default_device = all_devices[0];
    cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    Context context({ default_device });

    Program::Sources sources;

    // kernel calculates for each element C=A+B

    sources.push_back({ kernel_code.c_str(),kernel_code.length() });
    
    Program program(context, sources);
    if (program.build({ default_device }) != CL_SUCCESS) {
        cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }

    clock_t begin, end;
    double elapsed_secs = 0;
    long long int w_s; //palabras por segundo

    srand(time(NULL));//seed random

    string words = readFile("words.txt"); //leer palabras
    vector<string> l_words = strToVector(words); //libreria de palabras

    int long_text = 4000; //longitud de palabras por texto
    int total_n_text = 2560;

    printf("Num. textos: %i, Long Text: %i \n", total_n_text, long_text);


    int bach = 5;//
    int total_amount_words = 0;

    for (int i = 0; i < total_n_text/ bach; i++) {

        int n_text = bach; //cantidad de textos (documentos)

        unordered_map <string, int>* words_frec = new unordered_map <string, int>[n_text];
        // diccionario de frecuencias por palabra de cada documento

        for (std::size_t i = 0; i < n_text; i++) {
            words_frec[i] = get_frec(text_generator(l_words, long_text), long_text);
        }

        long amount_words = count_words(words_frec, n_text);
        total_amount_words += amount_words;

        //printf("Length words: %i words \n", amount_words);

        char* s_in = new char[amount_words * MAX_LEN_WORD];
        int* f_in = new int[amount_words];
        long* out = new long[amount_words * BITS];

        compress_sim_data_cuda(words_frec, s_in, f_in, n_text);

        begin = clock();

        // create buffers on the device
        Buffer cu_s_in(context, CL_MEM_READ_WRITE, amount_words * sizeof(char) * MAX_LEN_WORD);
        Buffer cu_f_in(context, CL_MEM_READ_WRITE, amount_words * sizeof(int));
        Buffer cu_out(context, CL_MEM_READ_WRITE, amount_words * sizeof(long) * BITS);
        Buffer cu_len(context, CL_MEM_READ_WRITE, sizeof(long));

        //create queue to which we will push commands for the device.
        CommandQueue queue(context, default_device);

        queue.enqueueWriteBuffer(cu_s_in, CL_TRUE, 0, amount_words * sizeof(char) * MAX_LEN_WORD, s_in);
        queue.enqueueWriteBuffer(cu_f_in, CL_TRUE, 0, amount_words * sizeof(int), f_in);
        queue.enqueueWriteBuffer(cu_len, CL_TRUE, 0, sizeof(long), &amount_words);

        //run the kernel
        Kernel sim_hash_opencl(program, "opencl_sim_hash");
        sim_hash_opencl.setArg(0, cu_s_in);
        sim_hash_opencl.setArg(1, cu_f_in);
        sim_hash_opencl.setArg(2, cu_out);
        sim_hash_opencl.setArg(3, cu_len);

        queue.enqueueNDRangeKernel(sim_hash_opencl, NullRange, NDRange(amount_words), NullRange);
        queue.finish();

        queue.enqueueReadBuffer(cu_out, CL_TRUE, 0, amount_words * sizeof(long) * BITS, out);

        end = clock();
        elapsed_secs += double(end - begin) / CLOCKS_PER_SEC;

        queue.flush();

        string* r_out = extract_sim_data_cuda(out, words_frec, n_text);
        /*
        for (std::size_t i = 0; i < n_text; i++) {
            cout << r_out[i] << endl;
        }*/

        delete s_in;
        delete f_in;
        delete out;
        //delete words_frec;
    }

    printf("Total Words:  %ld words \n", total_amount_words);
    printf("Tiempo OpenCL:  %f ms \n", elapsed_secs);
    w_s = total_amount_words / elapsed_secs;
    printf("palabras por Seg:  %d words \n", w_s);


    
    unordered_map <string, int>* words_frec = new unordered_map <string, int>[total_n_text];

    for (std::size_t i = 0; i < total_n_text; i++) {
        words_frec[i] = get_frec(text_generator(l_words, long_text), long_text);
    }

    long amount_words = count_words(words_frec, total_n_text);

    char* s_in_l = new char[amount_words * MAX_LEN_WORD];
    int* f_in_l = new int[amount_words];
    long* out_l = new long[amount_words * BITS];

    compress_sim_data_cuda(words_frec, s_in_l, f_in_l, total_n_text);

    begin = clock();

    sim_hash_lineal(s_in_l, f_in_l, out_l, amount_words);

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("Tiempo Lineal:  %f ms \n", elapsed_secs);
    w_s = amount_words / elapsed_secs;
    printf("palabras por Seg:  %d words \n", w_s);

    string* r_out_l = extract_sim_data_cuda(out_l, words_frec, total_n_text);

    return 0;
}
