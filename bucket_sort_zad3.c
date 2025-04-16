#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

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

    int* bucket_positions;
} BucketSortConfig;


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



void allocate_global_buckets(int buckets_amount, uint32_t*** global_buckets, uint32_t** global_bucket_lengths, int problem_size) {
    *global_buckets = calloc(buckets_amount, sizeof(uint32_t *));
    *global_bucket_lengths = calloc(buckets_amount, sizeof(uint32_t));
    for (int i = 0; i < buckets_amount; i++) {
        (*global_buckets)[i] = calloc(problem_size, sizeof(uint32_t));
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

void distribute_to_private_buckets(BucketSortConfig* config) {
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int start_index = config->problem_size / config->thread_num * thread_id;

        int end_index;
        if (thread_id == config->thread_num - 1) {
            end_index = config->problem_size;
        } else {
            end_index = config->problem_size / config->thread_num * (thread_id + 1);
        }
        for (int i = start_index; i < end_index; i++) {
            size_t bucket_index = get_bucket_index(config->main_array[i], config->min_value, config->max_value, config->buckets_amount);
            int len = config->private_bucket_lengths[thread_id][bucket_index];
            int cap = config->private_bucket_capacities[thread_id][bucket_index];

            // POWIĘKSZANIE DYNAMICZNE
            if (len >= cap) {
                cap *= 2;
                config->private_bucket_capacities[thread_id][bucket_index] = cap;
                config->private_buckets[thread_id][bucket_index] = realloc(config->private_buckets[thread_id][bucket_index], cap * sizeof(uint32_t));
                if (config->private_buckets[thread_id][bucket_index] == NULL) {
                    printf("realloc failed in bucket %ld by thread %d\n", bucket_index, thread_id);
                    exit(1);
                }
            }

            config->private_buckets[thread_id][bucket_index][len] = config->main_array[i];
            config->private_bucket_lengths[thread_id][bucket_index]++;
        }
    }
}


void merge_private_buckets(BucketSortConfig* config) {
    #pragma omp parallel for
    for (int i = 0; i < config->buckets_amount; i++) {
        int offset = 0;
        for (int t = 0; t < config->thread_num; t++) {
            int len = config->private_bucket_lengths[t][i];
            for (int j = 0; j < len; j++) {
                config->global_buckets[i][offset++] = config->private_buckets[t][i][j];
            }
        }
        config->global_bucket_lengths[i] = offset;
    }
}


int compare(const void *a, const void *b) {
    uint32_t val1 = *(const uint32_t *)a;
    uint32_t val2 = *(const uint32_t *)b;

    if (val1 < val2) return -1;
    if (val1 > val2) return 1;
    return 0;
}


bool is_sorted(uint32_t *tab, int n) {
    for (int i = 1; i < n; i++) {
        if (tab[i - 1] > tab[i]) {
            return false;
        }
    }
    return true;
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
    free(config->bucket_positions);
    free(config->main_array);
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

    omp_set_num_threads(config.thread_num);

    double main_start_time = omp_get_wtime();

    // drawing numbers
    double start_time = omp_get_wtime();

    benchmark(&config);

    double end_time = omp_get_wtime();
    printf("benchmark,%s,%d,%d,%d,%.6f,%d\n", config.schedule_type, config.chunk_size, config.problem_size, config.thread_num, end_time - start_time, config.buckets_amount);

    find_min_max(config.main_array, config.problem_size, &config.min_value, &config.max_value);

    // allocate buckets
    allocate_private_buckets(&config);
    allocate_global_buckets(config.buckets_amount, &config.global_buckets, &config.global_bucket_lengths, config.problem_size);

    // distribute to buckets and merge buckets
    start_time = omp_get_wtime();
    distribute_to_private_buckets(&config);
    merge_private_buckets(&config);
    end_time = omp_get_wtime();

    printf("distribution_merge,%s,%d,%d,%d,%.6f,%d\n", config.schedule_type, config.chunk_size, config.problem_size, config.thread_num, end_time - start_time, config.buckets_amount);

    // sort buckets
    start_time = omp_get_wtime();
    #pragma omp parallel for
    for (int b = 0; b < config.buckets_amount; b++) {
        qsort(config.global_buckets[b], config.global_bucket_lengths[b], sizeof(uint32_t), compare);
    }
    end_time = omp_get_wtime();
    printf("sort,%s,%d,%d,%d,%.6f,%d\n", config.schedule_type, config.chunk_size, config.problem_size, config.thread_num, end_time - start_time, config.buckets_amount);

    // rewrite to final array
    start_time = omp_get_wtime();
    config.bucket_positions = calloc(config.buckets_amount, sizeof(uint32_t));
    int cumulative_length = 0;
    for (int i = 0; i < config.buckets_amount; i++) {
        config.bucket_positions[i] = cumulative_length;
        cumulative_length += config.global_bucket_lengths[i];
    }

    #pragma omp parallel for
    for (int i = 0; i < config.buckets_amount; i++) {
        for (int j = 0; j < config.global_bucket_lengths[i]; j++) {
            config.main_array[config.bucket_positions[i] + j] = config.global_buckets[i][j];
        }
    }

    end_time = omp_get_wtime();
    printf("rewrite,%s,%d,%d,%d,%.6f,%d\n", config.schedule_type, config.chunk_size, config.problem_size, config.thread_num, end_time - start_time, config.buckets_amount);

    double main_end_time = omp_get_wtime();
 
    printf("whole_algorithm,%s,%d,%d,%d,%.6f,%d\n", config.schedule_type, config.chunk_size, config.problem_size, config.thread_num, main_end_time - main_start_time, config.buckets_amount);

    // check if array is sorted
    if (!is_sorted(config.main_array, config.problem_size)) {
        printf("Tablica NIE jest posortowana.\n");
        free_all_memory(&config);
        exit(-1);
    }

    free_all_memory(&config);
    return 0;
}
