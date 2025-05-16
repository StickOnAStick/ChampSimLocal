#pragma once

#define GLOBAL_HISTORY_TOKEN_BITS 24
#define IP_LSB_TOKEN_BITS 16

constexpr int PATH_HISTORY_TOKEN_BITS = 24;
constexpr int PATH_HISTORY_REG_SIZE   = 8;

constexpr int NUM_LOCAL_HISTORY_SCALES = 18;
constexpr std::array<uint16_t, NUM_LOCAL_HISTORY_SCALES> LOCAL_HISTORY_TABLE_SIZES = { 512, 16384, 512, 16384, 4096, 32768, 32768, 128, 32768, 128, 16384, 4096, 1024, 16384, 2048, 512, 256, 4096 };
constexpr std::array<uint8_t, NUM_LOCAL_HISTORY_SCALES>  LOCAL_HISTORY_TABLE_BITS  = { 8, 6, 9, 11, 4, 5, 2, 10, 2, 6, 7, 3, 1, 10, 1, 12, 8, 3 };

constexpr int    START_HISTORY = 3;
constexpr double ALPHA         = 1.405;

#define HIDDEN_SIZE 14
#define NUM_HEADS 7
#define DROPOUT_RATE 0.086
#define LEARNING_RATE 0.000895
