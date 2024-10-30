#include <catch.hpp>
#include "mocks.hpp"
#include "ooo_cpu.h"
#include "instr.h"
#include "register_allocator.h"

SCENARIO("The register allocation logic correctly reassigns physical register names.") {
  GIVEN("An empty RAT"){    
    constexpr int PHYSICALREGS = 128;
    RegisterAllocator ra{PHYSICALREGS};
    
    WHEN("A write and then a read occurs to the same logical register"){
      auto write1 = champsim::test::instruction_with_ip(0);
      write1.destination_registers.push_back(5);
      write1.destination_registers[0] = ra.rename_dest_register(write1.destination_registers[0], write1);

      auto read1 = champsim::test::instruction_with_ip(1);
      read1.source_registers.push_back(5);
      read1.source_registers[0] = ra.rename_src_register(read1.source_registers[0]);

      THEN("The dest. register of the first and the source register of the second instruction match."){
        REQUIRE(write1.destination_registers[0] == read1.source_registers[0]);

        AND_THEN("There are PHYSICALREGS-1 free physical registers."){
          REQUIRE(ra.count_free_registers() == PHYSICALREGS - 1);
        }
      }

      AND_WHEN("A write occurs to the same logical register"){
        auto write2 = champsim::test::instruction_with_ip(2);
          write2.destination_registers.push_back(5);
        write2.destination_registers[0] = ra.rename_dest_register(write2.destination_registers[0], write2);
        
        THEN("The destination physical register does not match the previously assigned physical register."){
          REQUIRE(write1.destination_registers[0] != write2.destination_registers[0]);
        }

        AND_WHEN("Another read from the same logical register occurs"){
          auto read2 = champsim::test::instruction_with_ip(3);
            read2.source_registers.push_back(5);
          read2.source_registers[0] = ra.rename_src_register(read2.source_registers[0]);

          THEN("The physical register source matches the newly assigned register."){
            REQUIRE(read2.source_registers[0] == write2.destination_registers[0]);

            AND_THEN("There are PHYSICALREGS-2 free physical registers."){
              REQUIRE(ra.count_free_registers() == PHYSICALREGS - 2);
            }
          }
        }
      }
    }

    WHEN("A read occurs to a logical register that has not been allocated"){
      auto read1 = champsim::test::instruction_with_ip(0);
      read1.source_registers.push_back(2);
      read1.source_registers[0] = ra.rename_src_register(read1.source_registers[0]);

      THEN("The read has no invalid (unready) register operands"){
        REQUIRE(ra.count_reg_dependencies(read1) == 0);
        
        AND_THEN("The read's source is physical register 0"){
          REQUIRE(read1.source_registers[0] == 0);
        }
      }
      
      AND_WHEN("A write and then a read occur on the same logical register"){
        auto write1 = champsim::test::instruction_with_ip(1);
        write1.destination_registers.push_back(2);
        write1.destination_registers[0] = ra.rename_dest_register(write1.destination_registers[0], write1);
        auto read2 = champsim::test::instruction_with_ip(0);
        read2.source_registers.push_back(2);
        read2.source_registers[0] = ra.rename_src_register(read2.source_registers[0]);

        THEN("The second read's source is physical register 1"){
          REQUIRE(read2.source_registers[0] == 1);
          AND_THEN("The second read is waiting on one register to become valid."){
            REQUIRE(ra.count_reg_dependencies(read2) == 1);
          }
        }
      }
    }

    WHEN("A write and then 500 reads from the same logical register"){
      auto write1 = champsim::test::instruction_with_ip(0);
      write1.destination_registers.push_back(3);
      write1.destination_registers[0] = ra.rename_dest_register(write1.destination_registers[0], write1);
      for (int i = 0; i < 500; ++i){
        auto read1 = champsim::test::instruction_with_ip(i+1);
        read1.source_registers.push_back(3);
        read1.source_registers[0] = ra.rename_src_register(read1.source_registers[0]);
        THEN("The source of the last read is physical register 0."){
          REQUIRE(read1.source_registers[0] == 0);
        }
      }
      THEN("Physical Register 0 is not in the list of free registers."){
        REQUIRE(ra.count_free_registers() == PHYSICALREGS - 1);
      }
      AND_WHEN("Another write to the same physical register occurs"){
        auto write2 = champsim::test::instruction_with_ip(501);
        write2.destination_registers.push_back(3);
        write2.destination_registers[0] = ra.rename_dest_register(write2.destination_registers[0], write2);
        THEN("The destination of the new write is physical register 1"){
          REQUIRE(write2.destination_registers[0] == 1);
        }
      }
    }

    WHEN("A long chain of RAW dependencies"){
      for (int i = 0; i < 100; ++i){
        auto inst = champsim::test::instruction_with_ip(i);
        inst.destination_registers.push_back(3);
        inst.source_registers.push_back(3);
        inst.source_registers[0] = ra.rename_src_register(inst.source_registers[0]);
        inst.destination_registers[0] = ra.rename_dest_register(inst.destination_registers[0], inst);

        THEN("The source of instruction n is physical register n"){
          REQUIRE(inst.source_registers[0] == i);
        }
      }
    }

    WHEN("An instruction of the form X0 = X1 operation X1"){
      auto inst = champsim::test::instruction_with_ip(0);
      inst.destination_registers.push_back(3);
      inst.source_registers.push_back(4);
      inst.source_registers.push_back(4);
      inst.source_registers[0] = ra.rename_src_register(inst.source_registers[0]);
      inst.source_registers[1] = ra.rename_src_register(inst.source_registers[1]);
      inst.destination_registers[0] = ra.rename_dest_register(inst.destination_registers[0], inst);
      THEN("The two sources of the instruction have the same physical register"){
        REQUIRE(inst.source_registers[0] == inst.source_registers[1]);
      }
      AND_WHEN("Ten repetitions of the instruction of the form X0 = X1 op X1"){
        for (int i = 0; i < 10; ++i){
          inst = champsim::test::instruction_with_ip(i+1);
          inst.destination_registers.push_back(3);
          inst.source_registers.push_back(4);
          inst.source_registers.push_back(4);
          inst.source_registers[0] = ra.rename_src_register(inst.source_registers[0]);
          inst.source_registers[1] = ra.rename_src_register(inst.source_registers[1]);
          inst.destination_registers[0] = ra.rename_dest_register(inst.destination_registers[0], inst);
          THEN("The two sources of the instruction have the same physical register"){
            REQUIRE(inst.source_registers[0] == inst.source_registers[1]);
          }
          THEN("The destination physical register of the nth repetition is n+2"){
            REQUIRE(inst.destination_registers[0] == i+2);
          }
          THEN("The source of the nth instruction is physical register 0"){
            REQUIRE(inst.source_registers[0] == 0);
          }
        }
      }
    }
  }
}

SCENARIO("The register allocator correctly recycles physical registers when no longer in use."){
  GIVEN("An empty RAT"){
    constexpr unsigned schedule_width = 128;
    constexpr unsigned schedule_latency = 1;

    do_nothing_MRC mock_L1I, mock_L1D;
    O3_CPU uut{champsim::core_builder{}
      .schedule_width(champsim::bandwidth::maximum_type{schedule_width})
      .schedule_latency(schedule_latency)
      .fetch_queues(&mock_L1I.queues)
      .data_queues(&mock_L1D.queues)
    };

    uut.ROB.push_back(champsim::test::instruction_with_ip(1));
    for (auto &instr : uut.ROB)
      instr.ready_time = champsim::chrono::clock::time_point{};

    constexpr int PHYSICALREGS = 128;
    RegisterAllocator ra{PHYSICALREGS};

    WHEN("A write and then a read on the same logical register are scheduled, but only the write executes"){
      auto write1 = champsim::test::instruction_with_ip(1);
      write1.destination_registers.push_back(5);
      write1.instr_id = 1;
      write1.destination_registers[0] = ra.rename_dest_register(write1.destination_registers[0], write1);

      auto read1 = champsim::test::instruction_with_ip(2);
      read1.source_registers.push_back(5);
      read1.instr_id = 2;
      read1.source_registers[0] = ra.rename_src_register(read1.source_registers[0]);
      
      ra.complete_dest_register(write1.destination_registers[0]);

      AND_WHEN("write1 retires"){
        write1.completed = true;
        ra.complete_dest_register(write1.destination_registers[0]);
        ra.retire_dest_register(write1.destination_registers[0]);
        THEN("No registers should have been recycled since no new writes to that arch reg"){
          REQUIRE(ra.count_free_registers() == PHYSICALREGS-1);
        }
        AND_WHEN("the read instruction is retired"){
          read1.completed = true;
          THEN("No registers should have been recycled since no new writes to that arch reg"){
            REQUIRE(ra.count_free_registers() == PHYSICALREGS-1);
          }
          AND_WHEN("A new write to the same register occurs"){
            auto write2 = champsim::test::instruction_with_ip(3);
            write2.destination_registers.push_back(5);
            write2.instr_id = 3;
            write2.destination_registers[0] = ra.rename_dest_register(write2.destination_registers[0], write2);
            THEN("there should be PHYSICALREGS-2 free registers"){
              REQUIRE(ra.count_free_registers() == PHYSICALREGS-2);
            }
            AND_WHEN("The second write completes execution"){
              ra.complete_dest_register(write2.destination_registers[0]);
              THEN("There should be PHYSICALREGS-2 free registers because the instruction has not retired"){
                REQUIRE(ra.count_free_registers() == PHYSICALREGS-2);
              }
              AND_WHEN("The second write has retired"){
                write2.completed = true;
                ra.retire_dest_register(write2.destination_registers[0]);
                THEN("Exactly one physical register should have been freed."){
                  REQUIRE(ra.count_free_registers() == PHYSICALREGS-1);
                }
              }
            }
          }
        }
      }
    }

    WHEN("Ten writes to the same logical register occur"){
      auto write1 = champsim::test::instruction_with_ip(1);
      write1.destination_registers.push_back(5);
      write1.instr_id = 1;
      auto write2 = champsim::test::instruction_with_ip(2);
      write2.destination_registers.push_back(5);
      write2.instr_id = 2;
      auto write3 = champsim::test::instruction_with_ip(3);
      write3.destination_registers.push_back(5);
      write3.instr_id = 3;
      auto write4 = champsim::test::instruction_with_ip(4);
      write4.destination_registers.push_back(5);
      write4.instr_id = 4;
      auto write5 = champsim::test::instruction_with_ip(5);
      write5.destination_registers.push_back(5);
      write5.instr_id = 5;
      auto write6 = champsim::test::instruction_with_ip(6);
      write6.destination_registers.push_back(5);
      write6.instr_id = 6;
      auto write7 = champsim::test::instruction_with_ip(7);
      write7.destination_registers.push_back(5);
      write7.instr_id = 7;
      auto write8 = champsim::test::instruction_with_ip(8);
      write8.destination_registers.push_back(5);
      write8.instr_id = 8;
      auto write9 = champsim::test::instruction_with_ip(9);
      write9.destination_registers.push_back(5);
      write9.instr_id = 9;
      auto write10 = champsim::test::instruction_with_ip(10);
      write10.destination_registers.push_back(5);
      write10.instr_id = 10;
      write1.destination_registers[0] = ra.rename_dest_register(write1.destination_registers[0], write1);
      write2.destination_registers[0] = ra.rename_dest_register(write2.destination_registers[0], write2);
      write3.destination_registers[0] = ra.rename_dest_register(write3.destination_registers[0], write3);
      write4.destination_registers[0] = ra.rename_dest_register(write4.destination_registers[0], write4);
      write5.destination_registers[0] = ra.rename_dest_register(write5.destination_registers[0], write5);
      write6.destination_registers[0] = ra.rename_dest_register(write6.destination_registers[0], write6);
      write7.destination_registers[0] = ra.rename_dest_register(write7.destination_registers[0], write7);
      write8.destination_registers[0] = ra.rename_dest_register(write8.destination_registers[0], write8);
      write9.destination_registers[0] = ra.rename_dest_register(write9.destination_registers[0], write9);
      write10.destination_registers[0] = ra.rename_dest_register(write10.destination_registers[0], write10);
      THEN("10 physical registers should be occupied"){
        REQUIRE(ra.count_free_registers() == PHYSICALREGS-10);
      }
      AND_WHEN("All ten writes complete and are then retired"){
        ra.retire_dest_register(write1.destination_registers[0]);
        ra.retire_dest_register(write2.destination_registers[0]);
        ra.retire_dest_register(write3.destination_registers[0]);
        ra.retire_dest_register(write4.destination_registers[0]);
        ra.retire_dest_register(write5.destination_registers[0]);
        ra.retire_dest_register(write6.destination_registers[0]);
        ra.retire_dest_register(write7.destination_registers[0]);
        ra.retire_dest_register(write8.destination_registers[0]);
        ra.retire_dest_register(write9.destination_registers[0]);
        ra.retire_dest_register(write10.destination_registers[0]);
        write1.completed = true;
        write2.completed = true;
        write3.completed = true;
        write4.completed = true;
        write5.completed = true;
        write6.completed = true;
        write7.completed = true;
        write8.completed = true;
        write9.completed = true;
        write10.completed = true;
        uut.operate();
        THEN("PHYSICALREGS-1 registers should be free"){
          REQUIRE(ra.count_free_registers() == PHYSICALREGS-1);
        }
      }
    }

    WHEN("Ten writes to different logical registers occur"){
      auto write1 = champsim::test::instruction_with_ip(1);
      write1.destination_registers.push_back(1);
      write1.instr_id = 1;
      auto write2 = champsim::test::instruction_with_ip(2);
      write2.destination_registers.push_back(2);
      write2.instr_id = 2;
      auto write3 = champsim::test::instruction_with_ip(3);
      write3.destination_registers.push_back(3);
      write3.instr_id = 3;
      auto write4 = champsim::test::instruction_with_ip(4);
      write4.destination_registers.push_back(4);
      write4.instr_id = 4;
      auto write5 = champsim::test::instruction_with_ip(5);
      write5.destination_registers.push_back(5);
      write5.instr_id = 5;
      auto write6 = champsim::test::instruction_with_ip(6);
      write6.destination_registers.push_back(6);
      write6.instr_id = 6;
      auto write7 = champsim::test::instruction_with_ip(7);
      write7.destination_registers.push_back(7);
      write7.instr_id = 7;
      auto write8 = champsim::test::instruction_with_ip(8);
      write8.destination_registers.push_back(8);
      write8.instr_id = 8;
      auto write9 = champsim::test::instruction_with_ip(9);
      write9.destination_registers.push_back(9);
      write9.instr_id = 9;
      auto write10 = champsim::test::instruction_with_ip(10);
      write10.destination_registers.push_back(10);
      write10.instr_id = 10;
      write1.destination_registers[0] = ra.rename_dest_register(write1.destination_registers[0], write1);
      write2.destination_registers[0] = ra.rename_dest_register(write2.destination_registers[0], write2);
      write3.destination_registers[0] = ra.rename_dest_register(write3.destination_registers[0], write3);
      write4.destination_registers[0] = ra.rename_dest_register(write4.destination_registers[0], write4);
      write5.destination_registers[0] = ra.rename_dest_register(write5.destination_registers[0], write5);
      write6.destination_registers[0] = ra.rename_dest_register(write6.destination_registers[0], write6);
      write7.destination_registers[0] = ra.rename_dest_register(write7.destination_registers[0], write7);
      write8.destination_registers[0] = ra.rename_dest_register(write8.destination_registers[0], write8);
      write9.destination_registers[0] = ra.rename_dest_register(write9.destination_registers[0], write9);
      write10.destination_registers[0] = ra.rename_dest_register(write10.destination_registers[0], write10);
      uut.operate();
      THEN("10 physical registers should be occupied"){
        REQUIRE(ra.count_free_registers() == PHYSICALREGS-10);
      }
      AND_WHEN("All ten writes are retired"){
        ra.retire_dest_register(write1.destination_registers[0]);
        ra.retire_dest_register(write2.destination_registers[0]);
        ra.retire_dest_register(write3.destination_registers[0]);
        ra.retire_dest_register(write4.destination_registers[0]);
        ra.retire_dest_register(write5.destination_registers[0]);
        ra.retire_dest_register(write6.destination_registers[0]);
        ra.retire_dest_register(write7.destination_registers[0]);
        ra.retire_dest_register(write8.destination_registers[0]);
        ra.retire_dest_register(write9.destination_registers[0]);
        ra.retire_dest_register(write10.destination_registers[0]);
        write1.completed = true;
        write2.completed = true;
        write3.completed = true;
        write4.completed = true;
        write5.completed = true;
        write6.completed = true;
        write7.completed = true;
        write8.completed = true;
        write9.completed = true;
        write10.completed = true;
        THEN("10 physical registers should be occupied"){
          REQUIRE(ra.count_free_registers() == PHYSICALREGS-10);
        }
      }
    }
  }
}
