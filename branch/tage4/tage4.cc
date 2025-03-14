
#include <map>
#include <bitset>
#include <iostream>
#include <string>
#include <math.h>
#include <random>

#include "msl/fwcounter.h"
#include "ooo_cpu.h"

#define Tag uint16_t
#define Index uint16_t
#define Path uint64_t
#define History uint64_t
#define BIMODAL_TABLE_SIZE 8192
#define MAX_INDEX_BITS 12
#define TAGE_TABLES 4
#define TAGE_TAG_BITS 7
#define TAGE_CONTER_BITS 3
#define TAGE_USEFUL_BITS 2
#define GLOBAL_HISTORY_LENGTH 1024
#define PATH_HISTORY_BUFFER_LENGTH 32
#define TAGE_MIN_LENGTH 5
#define HISTORY_ALPHA 1.5
#define TAGE_RESET_INTERVAL 512000
#define BIMODE_COUNTER_BITS  3

int debug_counter = 0;

struct TAGs
    {
        Tag tag;
        uint8_t useful;
        uint8_t counter;
    };


class Tage {
    private:
    int count;
    std::bitset<GLOBAL_HISTORY_LENGTH> GLOBAL_HISTORY;
    std::bitset<PATH_HISTORY_BUFFER_LENGTH> PATH_HISTORY;
    uint8_t T0[BIMODAL_TABLE_SIZE];
    std::array<TAGs, BIMODAL_TABLE_SIZE> T[TAGE_TABLES];
    int table_history_lengths[TAGE_TABLES];
    uint8_t use_alt_on_na;
    uint8_t tage_pred, pred, alt_pred;
    int pred_comp, alt_comp; // Provider and alternate component of last branch PC
    int STRONG;
    int debug_ct[7] = {0,0,0,0,0,0,0};

public:
    void init();  // initialise the member variables
    uint8_t predict(uint64_t ip);  // return the prediction from tage
    void update(uint64_t ip, uint8_t taken);  // updates the state of tage

    Index get_bimodal_index(uint64_t ip);   // helper hash function to index into the bimodal table
    Index get_predictor_index(uint64_t ip, int table);   // helper hash function to index into the predictor table using histories
    Tag get_tag(uint64_t ip, int table);   // helper hash function to get the tag of particular ip and table
    int get_match_below_n(uint64_t ip, int table);   // helper function to find the hit table strictly before the table argument
    void ctr_update(uint8_t &ctr, int cond, int low, int high);   // counter update helper function (including clipping)
    uint8_t get_prediction(uint64_t ip, int comp);   // helper function for prediction
    Path get_path_history_hash(int table);   // hepoer hash function to compress the path history
    History get_compressed_global_history(int inSize, int outSize); // Compress global history of last 'inSize' branches into 'outSize' by wrapping the history
};

uint8_t Tage::predict(uint64_t ip){


    pred_comp = get_match_below_n(ip,TAGE_TABLES + 1); // Get the first predictor from the end which matches the PC
    alt_comp = get_match_below_n(ip, pred_comp); // Get the first predictor below the provider which matches the PC 

    pred = get_prediction(ip, pred_comp);
    alt_pred = get_prediction(ip, alt_comp);

    // std::cout << debug_ct[0] << "|" << debug_ct[1] << "|" << debug_ct[2] << "|" << debug_ct[3] << "|" << "|" << debug_ct[4] << "|" << debug_ct[5] << std::endl;
    debug_ct[pred_comp]++;
    if(pred_comp == 0) { // if there is no alternate predictor we use the default bimodal table
        tage_pred = pred;
        //debug_ct[0]++;
    }
    else // if there is an alternate predictor 
    {
        Index index = get_predictor_index(ip,pred_comp);
        STRONG = abs(2*T[pred_comp-1][index].counter + 1 - (1 << TAGE_CONTER_BITS)) > 1; // check to see if the current predictors guess is strong
        // std::cout << "use_alt_on_na" << int(use_alt_on_na) << " | " << "STRONG:" << STRONG << std::endl;
        if (use_alt_on_na < 8 || STRONG) {
            tage_pred = pred; // if the prediction is strong, use that predictor
            //debug_ct[1]++;
        }
        else
        { 
            tage_pred = alt_pred; // if the prediction is not strong, use an alternate predictor 
        }
    }
    return tage_pred;
}

void Tage::ctr_update(uint8_t &ctr, int cond, int low, int high)
{

    if(cond && ctr < high)
        ctr++;
    else if (!cond && ctr > low)
        ctr--;
}

void Tage::update(uint64_t ip, uint8_t taken){
    if (pred_comp > 0)
    {
        struct TAGs *entry = &T[pred_comp-1][get_predictor_index(ip,pred_comp)];
        uint8_t useful = entry->useful;
        if(!STRONG)
        {
            if (pred != alt_pred)
                // std::cout << "ENTRY UPDATED" << std::endl;
                ctr_update(use_alt_on_na, !(pred = taken), 0, 15);
        }

        if (alt_comp > 0) 
        {
            struct TAGs *alt_entry = &T[alt_comp-1][get_predictor_index(ip,alt_comp)];
            if(useful == 0)
                // std::cout << "ENTRY UPDATED" << std::endl;
                ctr_update(alt_entry->counter,taken,0,((1 << TAGE_CONTER_BITS) -1));
        }

        else 
        {
            Index index = get_bimodal_index(ip);
            if (useful == 0)
                ctr_update(T0[index],taken,0,((1 << BIMODE_COUNTER_BITS)-1));
        }

        if(pred != alt_pred)
        {
            if (pred == taken)
            {
                if (entry->useful < ((1 << TAGE_USEFUL_BITS)-1))
                    entry->useful++;
            }
            else 
            {
                if(use_alt_on_na < 8)
                {
                    if (entry->useful > 0)
                        entry->useful--;
                }
            }
        }
        ctr_update(entry->counter, taken, 0, ((1 <<TAGE_CONTER_BITS) -1));
    }

    else
    {
        Index index = get_bimodal_index(ip);
        ctr_update(T0[index], taken, 0, ((1 << TAGE_CONTER_BITS)-1));
    }
    if (tage_pred != taken)
    {
        long random = static_cast <long> (rand()) / static_cast <long> (RAND_MAX);
        random = random & ((1 << (TAGE_TABLES - pred_comp -1))-1);
        int start_component = pred_comp + 1;

        if(random & 1)
        {
            start_component++;
            if(random & 2)
                start_component++;
        }
        int isFree = 0;
        for (int i = pred_comp + 1; i <= TAGE_TABLES; i++)
        {
            struct TAGs *entry_new = &T[i-1][get_predictor_index(ip,i)];
            if(entry_new->useful == 0)
                isFree = 1;
        }
        if (!isFree && start_component <= TAGE_TABLES)
            T[start_component-1][get_predictor_index(ip,start_component)].useful = 0;

        for(int i = start_component; i <= TAGE_TABLES; i++)
        {
            struct TAGs *entry_new = &T[i-1][get_predictor_index(ip,i)];
            if(entry_new->useful == 0)
            {
                entry_new->tag = get_tag(ip,i);
                entry_new->counter = (1 << (TAGE_CONTER_BITS - 1));
                break;
            }
        }
    }
    for(int i = GLOBAL_HISTORY_LENGTH - 1; i > 0; i--) 
        GLOBAL_HISTORY[i] = GLOBAL_HISTORY[i-1];
    GLOBAL_HISTORY[0] = taken;
    
    for(int i = PATH_HISTORY_BUFFER_LENGTH - 1; i > 0; i--)
        PATH_HISTORY[i] = PATH_HISTORY[i-1];
    PATH_HISTORY[0] = ip & 1;

    count++;
    if (count % TAGE_RESET_INTERVAL == 0)
    {
        count = 0;
        for(int i = 0; i < TAGE_TABLES; i++)
        {
            for (int j =0; j < (1 << MAX_INDEX_BITS); j++)
                T[i][j].useful >>= 1;
        }
    }
}

uint8_t Tage::get_prediction(uint64_t ip, int comp){
    if (comp == 0){
        Index index = get_bimodal_index(ip);
        // std::cout << "T0[index] = " << int(T0[index]) << std::endl;
        return (T0[index] >= (1 << (TAGE_CONTER_BITS-1)));
    }
    else 
    {
        Index index = get_predictor_index(ip,comp);
        // std::cout << "T[comp-1][index] = " << T[comp-1][index].counter << std::endl;
        return (T[comp-1][index].counter >= (1 << (TAGE_CONTER_BITS-1)));
    }
}

Index Tage::get_bimodal_index(uint64_t ip) {
    return ip & (BIMODAL_TABLE_SIZE - 1);
}

Tag Tage::get_tag(uint64_t ip, int table) {
    History global_history_hash = get_compressed_global_history(table_history_lengths[table-1],TAGE_TAG_BITS);
    global_history_hash ^= get_compressed_global_history(table_history_lengths[table-1],TAGE_TAG_BITS-1);
    return ((global_history_hash ^ ip) & ((1 << TAGE_TAG_BITS)-1));
}

int Tage::get_match_below_n(uint64_t ip, int table)
{
    for(int i = table - 1; i >= 1; i--){
        Index index = get_predictor_index(ip,i);
        Tag tag = get_tag(ip,i);
         
        if (T[i-1][index].tag == tag)
        {
            // std::cout<< T[i-1][index].tag << " | " << tag << "|" << i << std::endl ;
            return i;
        }
    }
    return 0;
}

void Tage::init()
{
    srand(time(0));
    use_alt_on_na = 8;
    tage_pred = 0;

    for (int i = 0; i < BIMODAL_TABLE_SIZE; i++)
        T0[i] = (1 << (BIMODE_COUNTER_BITS - 1)); // weakly taken
    for (int i = 0; i < TAGE_TABLES; i++){
        for (int j = 0; j < (1 << MAX_INDEX_BITS); j++)
        {
            T[i][j].counter = (1 << (BIMODE_COUNTER_BITS - 1));
            T[i][j].useful = 0;
            T[i][j].tag = 0;
        }
    }
    double power = 1;
    for (int i = 0; i < TAGE_TABLES; i++)
    {
        table_history_lengths[i] = int(TAGE_MIN_LENGTH * power + 0.5);
        power *= HISTORY_ALPHA;
        std::cout << "lengths" << table_history_lengths[i] << std::endl;
    }
    // for (int i = 0; i < TAGE_TABLES; i++)
    // {
    //     for (int j = 0; j < (1 << MAX_INDEX_BITS); j++)
    //         std::cout << T[i][j].tag << "|" << int(T[i][j].useful) << "|" << int(T[i][j].counter)  << std::endl;
    // }
    // exit(0);
    
}

Path Tage::get_path_history_hash(int table)
{
    Path A = 0; 
    Path size = table_history_lengths[table-1] > 16 ? 16 : table_history_lengths[table-1];
    for (int i = PATH_HISTORY_BUFFER_LENGTH -1; i>= 0; i--)
        A = (A << 1) | PATH_HISTORY[i];
    A = A & ((1 << size)-1);
    
    Path A1, A2;
    A1 = A & ((1 << MAX_INDEX_BITS)-1);
    A2 = A >> MAX_INDEX_BITS;

    // Use hashign from CBP-4 L-tage submission
    A2 = ((A2 << table) & ((1 << MAX_INDEX_BITS) - 1)) + (A2 >> abs(MAX_INDEX_BITS - table));
    A = A1 ^ A2;
    A = ((A << table) & ((1 << MAX_INDEX_BITS) - 1)) + (A >> abs(MAX_INDEX_BITS - table));
    return(A);
}

Index Tage::get_predictor_index(uint64_t ip, int table) {
    Path path_history_hash = get_path_history_hash(table);

    // Hash of global history
    History global_histor_hash = get_compressed_global_history(table_history_lengths[table-1],MAX_INDEX_BITS);

    // Really complex hashing function 
    return(global_histor_hash ^ ip ^ (ip >> (abs(MAX_INDEX_BITS-table)+1)) ^ path_history_hash) & ((1 << MAX_INDEX_BITS)-1);
}

History Tage::get_compressed_global_history(int inSize, int outSize){
History compressed_history = 0;
History temporary_history = 0;
int compressed_history_length = outSize;
for (int i = 0; i < inSize; i++)
{
    if (i % compressed_history_length == 0)
    {
        compressed_history ^= temporary_history;
        temporary_history = 0;
    }
    temporary_history = (temporary_history << 1) | GLOBAL_HISTORY[i];
}
compressed_history ^= temporary_history;
return compressed_history;
}

Tage tage_predictor[NUM_CPUS];

void O3_CPU::initialize_branch_predictor() {
    tage_predictor[cpu].init();

}

uint8_t O3_CPU::predict_branch(uint64_t ip)
{  
    //std::cout << debug_counter << std::endl;
    return tage_predictor[cpu].predict(ip);
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type)
{
    tage_predictor[cpu].update(ip,taken);
}