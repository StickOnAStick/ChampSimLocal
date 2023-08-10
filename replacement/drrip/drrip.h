#ifndef REPLACEMENT_DRRIP_H
#define REPLACEMENT_DRRIP_H

#include <array>
#include <vector>

#include "cache.h"
#include "modules.h"
#include "msl/fwcounter.h"

struct drrip : champsim::modules::replacement {
  private:
  unsigned& get_rrpv(long set, long way);

  public:
  static constexpr unsigned maxRRPV = 3;
  static constexpr std::size_t NUM_POLICY = 2;
  static constexpr std::size_t SDM_SIZE = 32;
  static constexpr std::size_t TOTAL_SDM_SETS = NUM_CPUS * NUM_POLICY * SDM_SIZE;
  static constexpr unsigned BIP_MAX = 32;
  static constexpr unsigned PSEL_WIDTH = 10;

  long NUM_SET, NUM_WAY;

  unsigned bip_counter;
  std::vector<std::size_t> rand_sets;
  std::array<champsim::msl::fwcounter<PSEL_WIDTH>, NUM_CPUS> PSEL;
  std::vector<unsigned> rrpv;

  drrip(CACHE* cache);

  // void initialize_replacement()
  long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const CACHE::BLOCK* current_set, uint64_t ip, uint64_t full_addr, access_type type);
  void update_replacement_state(uint32_t triggering_cpu, long set, long way, uint64_t full_addr, uint64_t ip, uint64_t victim_addr, access_type type,
                                uint8_t hit);

  // use this function to print out your own stats at the end of simulation
  // void replacement_final_stats() {}

  void update_bip(long set, long way);
  void update_srrip(long set, long way);
};

#endif
