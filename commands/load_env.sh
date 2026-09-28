# 1. Start clean and load your Python module
module purge
module load Python/3.10.15

# 2. Create a hidden folder in your home directory to hold our hack
mkdir -p ~/.lib_hack

# 3. Create a symlink (shortcut) that points the file you HAVE to the name Python WANTS
ln -sf /usr/lib/x86_64-linux-gnu/libcrypt.so.1 ~/.lib_hack/libcrypt.so.2

# 4. Tell the system to check your hack folder first when looking for libraries
export LD_LIBRARY_PATH=~/.lib_hack:$LD_LIBRARY_PATH

# 5. Activate your environment and test it
source venv/bin/activate
python --version
