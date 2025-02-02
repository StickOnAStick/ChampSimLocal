ROOT_DIR = $(patsubst %/,%,$(dir $(abspath $(firstword $(MAKEFILE_LIST)))))

CPPFLAGS += -MMD -I$(ROOT_DIR)/inc
CXXFLAGS += --std=c++17 -Ofast -Wall -Wextra -Wshadow -Wpedantic -fexpensive-optimizations

# CUDA flags
CUDA_FLAGS = -arch=all-major -O3  -ltoir -gen-opt-lto --use_fast_math --cudadevrt static --prec-div=false --extra-device-vectorization --default-stream per-thread 
# Compiler and linker
CXX = g++
NVCC = nvcc

# vcpkg integration
TRIPLET_DIR = $(patsubst %/,%,$(firstword $(filter-out $(ROOT_DIR)/vcpkg_installed/vcpkg/, $(wildcard $(ROOT_DIR)/vcpkg_installed/*/))))
CPPFLAGS += -isystem $(TRIPLET_DIR)/include
LDLIBS  += -L$(TRIPLET_DIR)/lib -L$(TRIPLET_DIR)/lib/manual-link
LDLIBS   += -llzma -lz -lbz2 -lfmt -lpthread -lcudart -lcublas

.phony: all all_execs clean configclean test makedirs cuda.o

test_main_name=$(ROOT_DIR)/test/bin/000-test-main

all: cuda.o all_execs 

# Generated configuration makefile contains:
#  - $(executable_name), the list of all executables in the configuration
#  - $(build_dirs), the list of all directories that hold executables
#  - $(build_objs), the list of all object files corresponding to core sources
#  - $(module_dirs), the list of all directories that hold module object files
#  - $(module_objs), the list of all object files corresponding to modules
#  - All dependencies and flags assigned according to the modules
include _configuration.mk

all_execs: $(filter-out $(test_main_name), $(executable_name))

# Remove all intermediate files
clean:
	@-find src test .csconfig $(module_dirs) \( -name '*.o' -o -name '*.d' \) -delete &> /dev/null
	@-$(RM) inc/champsim_constants.h
	@-$(RM) inc/cache_modules.h
	@-$(RM) inc/ooo_cpu_modules.h
	@-$(RM) src/core_inst.cc
	@-$(RM) cuda.o
	@-$(RM) $(test_main_name)

# Remove all configuration files
configclean: clean
	@-$(RM) -r $(module_dirs) _configuration.mk

# Make directories that don't exist
# exclude "test" to not conflict with the phony target
$(filter-out test, $(sort $(build_dirs) $(module_dirs))): | $(dir $@)
	-mkdir $@

# All .o files should be made like .cc files
$(build_objs) $(module_objs):
	$(COMPILE.cc) $(OUTPUT_OPTION) $<

cuda.o:
	$(NVCC) -c $(ROOT_DIR)/inc/cuda.cu -o $@ $(CUDA_FLAGS)

# Add address sanitizers for tests
#$(test_main_name): CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer
$(test_main_name): CXXFLAGS += -g3 -Og -Wconversion
$(test_main_name): LDLIBS   += -lCatch2Main -lCatch2


# Link main executables
$(filter-out $(test_main_name), $(executable_name)):
	$(CXX) $^  cuda.o $(LDLIBS) -o $@

# Tests: build and run
test: $(test_main_name)
	$(test_main_name)

pytest:
	PYTHONPATH=$(PYTHONPATH):$(shell pwd) python3 -m unittest discover -v --start-directory='test/python'

-include $(foreach dir,$(wildcard .csconfig/*/) $(wildcard .csconfig/test/*/),$(wildcard $(dir)/obj/*.d))