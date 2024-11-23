#!/bin/bash

# Install necessary dependencies using apt-get
echo "Installing necessary dependencies..."
sudo apt-get update && sudo apt-get install -y \
    bison autoconf automake autoconf-archive meson ninja-build \
    libx11-dev libxft-dev libxext-dev libtool pkg-config \
    liblz4-dev liblzma-dev libzstd-dev libarchive-dev libxtst-dev libxrandr-dev

if [ $? -ne 0 ]; then
    echo "Failed to install required dependencies. Exiting."
    exit 1
fi

# Install the Jinja2 Python package
echo "Installing Jinja2 for Python3..."
python3 -m pip install --user jinja2

if [ $? -ne 0 ]; then
    echo "Failed to install Jinja2. Exiting."
    exit 1
fi

# Update the vcpkg.json file
echo "Updating the vcpkg.json file..."
VCPKG_JSON="vcpkg.json"
if [ -f "$VCPKG_JSON" ]; then
    # Use sed to insert "libtorch" into the dependencies section
    sed -i '/"dependencies".*{/{N;s/\({[^}]*\)\(.*\)/\1, "libtorch"\2/}' "$VCPKG_JSON"
    echo "Updated vcpkg.json successfully."
else
    echo "vcpkg.json not found. Exiting."
    exit 1
fi

# Re-run bootstrap.sh
echo "Running bootstrap-vcpkg.sh..."
if [ -f "vcpkg/bootstrap-vcpkg.sh" ]; then
    bash vcpkg/bootstrap-vcpkg.sh
else
    echo "vcpkg/bootstrap-vcpkg.sh not found. Exiting."
    exit 1
fi

# Install dependencies using vcpkg
echo "Installing dependencies using vcpkg..."
if [ -f "vcpkg/vcpkg" ]; then
    ./vcpkg/vcpkg install
else
    echo "vcpkg executable not found. Exiting."
    exit 1
fi

# Make the entire project
echo "Building the project using make..."
if [ -f "Makefile" ]; then
    make
else
    echo "Makefile not found. Exiting."
    exit 1
fi

echo "Script completed successfully."
