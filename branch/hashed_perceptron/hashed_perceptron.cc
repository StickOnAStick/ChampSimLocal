#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ooo_cpu.h"

#define NUM_TABLES 16
#define MAX_HISTORY 232
#define MIN_HISTORY 3
#define ADJUST_SPEED 18
#define LOG_TABLE_SIZE 12
#define TABLE_SIZE (1 << LOG_TABLE_SIZE)
#define HISTORY_WORDS (MAX_HISTORY / LOG_TABLE_SIZE + 1)

namespace {
    inline constexpr int historyLengths[NUM_TABLES] = {0, 3, 4, 6, 8, 10, 14, 19, 26, 36, 49, 67, 91, 125, 170, MAX_HISTORY};
    int weightTables[NUM_CPUS][NUM_TABLES][TABLE_SIZE];
    unsigned int globalHistory[NUM_CPUS][HISTORY_WORDS];
    uint64_t tableIndices[NUM_CPUS][NUM_TABLES];
    int predictionThreshold[NUM_CPUS], thresholdCounter[NUM_CPUS], perceptronSum[NUM_CPUS];
}

void O3_CPU::initialize_branch_predictor() {
    memset(::weightTables, 0, sizeof(::weightTables));
    memset(::globalHistory, 0, sizeof(::globalHistory));
    for (unsigned i = 0; i < NUM_CPUS; i++) {
        ::predictionThreshold[i] = 10;
    }
}

uint8_t O3_CPU::predict_branch(uint64_t ip) {
    ::perceptronSum[cpu] = 0;
    
    for (int i = 0; i < NUM_TABLES; i++) {
        int historyLength = historyLengths[i];
        uint64_t historyHash = 0;
        int fullWords = historyLength / LOG_TABLE_SIZE;
        int remainingBits = historyLength % LOG_TABLE_SIZE;
        
        for (int j = 0; j < fullWords; j++)
            historyHash ^= ::globalHistory[cpu][j];
        historyHash ^= ::globalHistory[cpu][fullWords] & ((1 << remainingBits) - 1);
        historyHash ^= ip;
        historyHash &= TABLE_SIZE - 1;
        
        ::tableIndices[cpu][i] = historyHash;
        ::perceptronSum[cpu] += ::weightTables[cpu][i][historyHash];
    }
    return ::perceptronSum[cpu] >= 1;
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type) {
    bool correctPrediction = taken == (::perceptronSum[cpu] >= 1);
    bool shiftBit = taken;
    
    for (int i = 0; i < HISTORY_WORDS; i++) {
        ::globalHistory[cpu][i] <<= 1;
        ::globalHistory[cpu][i] |= shiftBit;
        shiftBit = !!(::globalHistory[cpu][i] & TABLE_SIZE);
        ::globalHistory[cpu][i] &= TABLE_SIZE - 1;
    }
    
    int absPerceptronSum = (::perceptronSum[cpu] < 0) ? -::perceptronSum[cpu] : ::perceptronSum[cpu];
    
    if (!correctPrediction || absPerceptronSum < ::predictionThreshold[cpu]) {
        for (int i = 0; i < NUM_TABLES; i++) {
            int* weight = &::weightTables[cpu][i][::tableIndices[cpu][i]];
            if (taken) {
                if (*weight < 127) (*weight)++;
            } else {
                if (*weight > -128) (*weight)--;
            }
        }
        
        if (!correctPrediction) {
            ::thresholdCounter[cpu]++;
            if (::thresholdCounter[cpu] >= ADJUST_SPEED) {
                ::predictionThreshold[cpu]++;
                ::thresholdCounter[cpu] = 0;
            }
        } else if (absPerceptronSum < ::predictionThreshold[cpu]) {
            ::thresholdCounter[cpu]--;
            if (::thresholdCounter[cpu] <= -ADJUST_SPEED) {
                ::predictionThreshold[cpu]--;
                ::thresholdCounter[cpu] = 0;
            }
        }
    }
}
