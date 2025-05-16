#pragma once

#define GLOBAL_HISTORY_TOKEN_BITS 8
#define IP_LSB_TOKEN_BITS 12

constexpr int PATH_HISTORY_TOKEN_BITS = 12;
constexpr int PATH_HISTORY_REG_SIZE   = 4;

constexpr int NUM_LOCAL_HISTORY_SCALES = 8;
constexpr std::array<uint16_t, NUM_LOCAL_HISTORY_SCALES> LOCAL_HISTORY_TABLE_SIZES = { 4096, 4096, 16384, 16384, 256, 32768, 256, 16384 };
constexpr std::array<uint8_t, NUM_LOCAL_HISTORY_SCALES>  LOCAL_HISTORY_TABLE_BITS  = { 9, 9, 2, 1, 2, 1, 12, 2 };

constexpr int    START_HISTORY = 5;
constexpr double ALPHA         = 1.565;

#define HIDDEN_SIZE 18
#define NUM_HEADS 3
#define DROPOUT_RATE 0.138
#define LEARNING_RATE 0.001136