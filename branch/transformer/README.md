To use this branch predictor you will need to install some dependencies. 

### Pre-Req install APT depts
Install wsl dependencies: 
```bash
sudo apt-get install bison autoconf automake autoconf-archive meson ninja-build libx11-dev libxft-dev libxext-dev libtool pkg-config liblz4-dev liblzma-dev libzstd-dev libarchive-dev libxtst-dev libxrandr-dev
```

and
```bash
python -m pip install jinja2
```


### Update the manifest
Update the `vcpkg.json` config file. 

```json
"dependencies" : {
    ...,
    "libtorch" // Add this
 }
```

### Just in-case
Re-run `bootstrap.sh`

### Update make file:
To ensure that pytorch (libtorch) is properly linked we need to add the following libs and update the cpp paths to the make file: 
```make
LDLIBS   += -llzma -lz -lbz2 -lfmt -ltorch -ltorch_cpu -lc10
CPPFLAGS += -MMD -isystem $(TRIPLET_DIR)/include -isystem $(TRIPLET_DIR)/include/torch/csrc/api/include
```

**DONT** just copy paste these in anywhere, make sure you overwrite the existing `CPPFLAGS` and `LDLIBS` variables.


### Finally
Install the c++ dependencies:

```bash
sudo vcpkg/vcpkg install
```


# Setup vscode

It's highly likely that you're going to run into intellisense errors. 

## Ensure CMAKE is configured
CMake is required for the libtorch dependency. We need to make sure we have the __CMakeTools extension__ installed. 
Configure CMake to use your wsl __gcc__. Search Config: `CTRL + SHIFT + P` Select `x64-linux` from the menu. 


## __NOTE:__
This will take a LONG time to build. This is because PyTorch has a lot of dependencies to function.

Incase of a build failure, most of the time you are just missing a dependency for `apt` which will be made aware to you by reviewing the last bit of stack trace. 

If this is the case, install the dependency and add it to __Step 1__ of this file.  