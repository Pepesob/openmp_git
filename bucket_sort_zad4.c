#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

// makro do obliczania indeksu kubełka dla danej liczby
#define BUCKET_INDEX(number, min_value, max_value, number_of_buckets) \
    (((number - min_value) / (double)(max_value - min_value + 1)) * (number_of_buckets))

typedef struct {
    char* schedule_type;
    int chunk_size;
    int problem_size;
    int thread_num;
    int buckets_amount;

    uint32_t min_value;
    uint32_t max_value;

    uint32_t* main_array;

    uint32_t*** private_buckets;
    uint32_t** private_bucket_lengths;
    uint32_t** private_bucket_capacities;

    uint32_t** global_buckets;
    uint32_t* global_bucket_lengths;

    int* bucket_offsets;
} BucketSortConfig;

int compare(const void *a, const void *b) {
    uint32_t val1 = *(const uint32_t *)a;
    uint32_t val2 = *(const uint32_t *)b;

    if (val1 < val2) return -1;
    if (val1 > val2) return 1;
    return 0;
}


void random_num(int thread_num, uint32_t* tab, long long len) {
    long rand_value;
    struct drand48_data rand_buffer;
    srand48_r(thread_num, &rand_buffer);

    for (long long i = 0; i < len; i++) {
        lrand48_r(&rand_buffer, &rand_value);
        tab[i] = (uint32_t) rand_value;
    }
}

void benchmark(BucketSortConfig* config) {
    long long size_per_thread = config->problem_size / config->thread_num;
    long long last_size = config->problem_size - ((config->thread_num - 1) * size_per_thread);

    if (config->main_array == NULL) {
        printf("Calloc returned NULL!\n");
        exit(-1);
    }

    omp_sched_t schedule;
    if (strcmp(config->schedule_type, "static") == 0) schedule = omp_sched_static;
    else if (strcmp(config->schedule_type, "dynamic") == 0) schedule = omp_sched_dynamic;
    else if (strcmp(config->schedule_type, "guided") == 0) schedule = omp_sched_guided;
    else schedule = omp_sched_auto;

    omp_set_schedule(schedule, config->chunk_size);

    #pragma omp parallel
    {
    #pragma omp for schedule(runtime)
        for (int i = 0; i < config->thread_num; i++) {
            if (i == config->thread_num - 1)
                random_num(i, config->main_array + i * size_per_thread, last_size);
            else
                random_num(i, config->main_array + i * size_per_thread, size_per_thread);
        }
    }
}


void find_min_max(uint32_t* tab, int size, uint32_t* min_value, uint32_t* max_value) {
    *min_value = tab[0];
    *max_value = tab[0];

    for (int i = 1; i < size; i++) {
        if (tab[i] > *max_value)
            *max_value = tab[i];
        if (tab[i] < *min_value)
            *min_value = tab[i];
    }
}

void allocate_global_buckets(BucketSortConfig* config) {
    config->global_buckets = malloc(config->buckets_amount * sizeof(uint32_t *));
    config->global_bucket_lengths = calloc(config->buckets_amount, sizeof(uint32_t));
    for (int i = 0; i < config->buckets_amount; i++) {
        config->global_buckets[i] = calloc(config->problem_size, sizeof(uint32_t));
    }
}

void allocate_private_buckets(BucketSortConfig* config) {
    config->private_buckets = calloc(config->thread_num, sizeof(uint32_t **));
    config->private_bucket_lengths = calloc(config->thread_num, sizeof(uint32_t *));
    config->private_bucket_capacities = calloc(config->thread_num, sizeof(uint32_t *));

    int capacity = (config->problem_size / config->buckets_amount / config->thread_num);

    for (int t = 0; t < config->thread_num; t++) {
        config->private_buckets[t] = calloc(config->buckets_amount, sizeof(uint32_t *));
        config->private_bucket_lengths[t] = calloc(config->buckets_amount, sizeof(uint32_t));
        config->private_bucket_capacities[t] = calloc(config->buckets_amount, sizeof(uint32_t));
        for (int b = 0; b < config->buckets_amount; b++) {
            config->private_bucket_capacities[t][b] = capacity;
            config->private_buckets[t][b] = calloc(capacity , sizeof(uint32_t));
        }
    }
}

size_t get_bucket_index(uint32_t number, uint32_t min_value, uint32_t max_value, size_t bucket_count) {
    if (max_value == min_value) {
        return 0;
    }

    double normalized = (double)(number - min_value) / (max_value - min_value + 1);
    size_t index = (size_t)(normalized * bucket_count);

    if (index >= bucket_count) {
        index = bucket_count - 1;
    }

    return index;
}



void free_all_memory(BucketSortConfig* config) {
    for (int i = 0; i < config->thread_num; i++) {
        for (int j = 0; j < config->buckets_amount; j++) {
            free(config->private_buckets[i][j]);
        }
        free(config->private_buckets[i]);
        free(config->private_bucket_lengths[i]);
    }

    free(config->private_buckets);
    free(config->private_bucket_lengths);

    for (int i = 0; i < config->buckets_amount; i++) {
        free(config->global_buckets[i]);
    }

    free(config->global_buckets);
    free(config->global_bucket_lengths);
    free(config->bucket_offsets);
    free(config->main_array);
}


// zad 4
void distribute_to_global_buckets_(uint32_t* tab, long long tab_size, uint32_t** public_buckets, uint32_t* public_buckets_lengths,
    uint32_t min_value, uint32_t max_value, int buckets_amount) {
    for (long long i = 0; i < tab_size; i++) {
        size_t bucket_index = get_bucket_index(tab[i], min_value, max_value, buckets_amount);
        uint32_t pos;
        #pragma omp atomic capture
        pos = public_buckets_lengths[bucket_index]++;
        public_buckets[bucket_index][pos] = tab[i];
    }
}

void distribute_to_global_buckets(BucketSortConfig* config) {
    long long size_per_thread = config->problem_size / config->thread_num;
    long long last_size = config->problem_size - ((config->thread_num - 1) * size_per_thread);
    #pragma omp parallel for
    for (int i = 0; i < config->thread_num; i++) {
        long long tab_offset_size = (i == config->thread_num - 1) ? last_size : size_per_thread;
        distribute_to_global_buckets_(config->main_array + i * size_per_thread, tab_offset_size, config->global_buckets, config->global_bucket_lengths, config->min_value, config->max_value, config->buckets_amount);
    }
}



void sort_buckets(BucketSortConfig* config) {
    #pragma omp parallel for
    for (int b = 0; b < config->buckets_amount; b++)
    {
        qsort(config->global_buckets[b], config->global_bucket_lengths[b], sizeof(uint32_t), compare);
    }
}



void allocate_and_calculate_bucket_offsets(BucketSortConfig* config) {
    config->bucket_offsets = calloc(config->buckets_amount, sizeof(uint32_t));
    uint32_t prefix_sum = 0;
    for (int i = 0; i < config->buckets_amount; i++) {
        config->bucket_offsets[i] = prefix_sum;
        prefix_sum += config->global_bucket_lengths[i];
    }
}


void write_to_end_array_(uint32_t* tab, uint32_t public_bucket_size, uint32_t* public_bucket) {
    for (long long i=0; i<public_bucket_size; i++){
        tab[i] = public_bucket[i];
    }
}

void write_to_end_array(BucketSortConfig* config){
    allocate_and_calculate_bucket_offsets(config);
    #pragma omp parallel for
    for (long long i=0; i<config->buckets_amount; i++) {
        write_to_end_array_(config->main_array + config->bucket_offsets[i], config->global_bucket_lengths[i], config->global_buckets[i]);
    }
}

bool is_sorted(uint32_t *tab, int n) {
    for (int i = 1; i < n; i++) {
        if (tab[i - 1] > tab[i]) {
            return false;
        }
    }
    return true;
}



int main(int argc, char *argv[])
{
    if (argc != 6){
        printf("%d\n",argc);
        printf("Signature <schedule_type> <chunk_size> <problem_size> <thread_number> <number of buckets>\n");
        return 0;
    }

    BucketSortConfig config;
    config.schedule_type = argv[1];
    config.chunk_size = atoi(argv[2]);
    config.problem_size = (int) strtod(argv[3], NULL);
    config.thread_num = atoi(argv[4]);
    config.buckets_amount = atoi(argv[5]);
    config.main_array = calloc(config.problem_size, sizeof(uint32_t));

    char benchmark_buff[200];
    char distribution_buff[200];
    char sort_buff[200];
    char rewrite_buff[200];
    char wholealg_buff[200];
    char buckets_buff[200];

    omp_set_num_threads(config.thread_num);

    double main_start_time = omp_get_wtime();

    // drawing numbers
    double start_time = omp_get_wtime();

    benchmark(&config);

    double end_time = omp_get_wtime();
    sprintf(benchmark_buff, "benchmark,%s,%d,%d,%d,%.6f,%d", config.schedule_type, config.chunk_size, config.problem_size, config.thread_num, end_time - start_time, config.buckets_amount);


    find_min_max(config.main_array, config.problem_size, &config.min_value, &config.max_value);

    // allocate buckets
    allocate_private_buckets(&config);
    allocate_global_buckets(&config);

    // distribute to buckets and merge buckets
    start_time = omp_get_wtime();
    distribute_to_global_buckets(&config);
    end_time = omp_get_wtime();

    sprintf(distribution_buff, "distribution,%s,%d,%d,%d,%.6f,%d", config.schedule_type, config.chunk_size, config.problem_size, config.thread_num, end_time - start_time, config.buckets_amount);

    start_time = omp_get_wtime();
    sort_buckets(&config);
    end_time = omp_get_wtime();

    sprintf(sort_buff, "sort,%s,%d,%d,%d,%.6f,%d", config.schedule_type, config.chunk_size, config.problem_size, config.thread_num, end_time - start_time, config.buckets_amount);

    start_time = omp_get_wtime();
    write_to_end_array(&config);
    end_time = omp_get_wtime();

    double main_end_time = omp_get_wtime();

    sprintf(rewrite_buff, "rewrite,%s,%d,%d,%d,%.6f,%d", config.schedule_type, config.chunk_size, config.problem_size, config.thread_num, end_time - start_time, config.buckets_amount);
    sprintf(wholealg_buff, "whole_algorithm,%s,%d,%d,%d,%.6f,%d", config.schedule_type, config.chunk_size, config.problem_size, config.thread_num, main_end_time - main_start_time, config.buckets_amount);

    printf("{\n");
    printf("\"csv_vals\": [\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"],\n", benchmark_buff, distribution_buff, sort_buff, rewrite_buff, wholealg_buff);
    printf("\"buckets\": [");
    for (int i = 0; i < config.buckets_amount; i++) {
        if (i == config.buckets_amount-1) printf("%u", config.global_bucket_lengths[i]);
        else printf("%u,", config.global_bucket_lengths[i]);
    }
    printf("]\n");
    printf("}");

    // for (int i=0; i<config.problem_size; i++) {
    //     printf("%u\n", config.main_array[i]);
    // }
    // check if array is sorted
    if (!is_sorted(config.main_array, config.problem_size)) {
        printf("Tablica NIE jest posortowana.\n");
        free_all_memory(&config);
    }

    free_all_memory(&config);
    return 0;
}