#pragma once

#define GLOBAL_HISTORY_TOKEN_BITS 24
#define IP_LSB_TOKEN_BITS 16

constexpr int PATH_HISTORY_TOKEN_BITS = 24;
constexpr int PATH_HISTORY_REG_SIZE   = 8;

constexpr int NUM_LOCAL_HISTORY_SCALES = 6;
constexpr std::array<uint16_t, NUM_LOCAL_HISTORY_SCALES> LOCAL_HISTORY_TABLE_SIZES = { 2048, 2048, 256, 4096, 256, 16384 };
constexpr std::array<uint8_t, NUM_LOCAL_HISTORY_SCALES>  LOCAL_HISTORY_TABLE_BITS  = { 5, 2, 5, 3, 4, 1 };

constexpr int    START_HISTORY = 5;
constexpr double ALPHA         = 1.265;

#define HIDDEN_SIZE 6
#define NUM_HEADS 2
#define DROPOUT_RATE 0.174
// #define LEARNING_RATE 0.00001
#define LEARNING_RATE 0.001064
