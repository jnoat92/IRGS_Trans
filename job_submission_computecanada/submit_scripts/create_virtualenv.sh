
module load python/3.9.6
module load cmake

echo "loading module done"

echo "Creating new virtualenv"

virtualenv --no-download ~/torch_magic1
source ~/torch_magic1/bin/activate

echo "Virtual environment creation done"

pip install --no-index --upgrade pip
pip install torch torchvision torchaudio
pip install albumentations
pip install h5py
pip install icecream
pip install joblib
pip install matplotlib
pip install mycolorpy
pip install numba
pip install numpy
#pip install opencv-python
pip install pandas
pip install pillow
pip install scikit-image
pip install scikit-learn
pip install timm
pip install tqdm
pip install wandb

pip install xarray==2022.10.0 h5netcdf

# #MAGIC LIB
echo "Installing Magic lib"
cd ..
cd ..
cd ..
git clone --recurse-submodules https://github.com/Max-Manning/magic_lib 
cd magic_lib
python setup.py install


echo "virtual env creation done"