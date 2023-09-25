export OMP_NUM_THREADS=8
num_gpu="4"
num_steps="10"

#########################################################################
# Sampling with DDIM
#########################################################################
sampler="ddim"
outdir="./fid-tmp/cifar10/"$sampler"_"$num_steps
torchrun --standalone --nproc_per_node=$num_gpu generate.py --outdir=$outdir --seeds=0-49999 --subdirs \
    --sampler_type=$sampler --steps=$num_steps --batch=100 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

torchrun --standalone --nproc_per_node=$num_gpu fid.py calc --images=$outdir \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz


sampler="d_ddim"
outdir="./fid-tmp/cifar10/"$sampler"_"$num_steps
torchrun --standalone --nproc_per_node=$num_gpu generate.py --outdir=$outdir --seeds=0-49999 --subdirs \
    --sampler_type=$sampler --steps=$num_steps --batch=100 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

torchrun --standalone --nproc_per_node=$num_gpu fid.py calc --images=$outdir \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz


#########################################################################
# Sampling with EDM
#########################################################################

sampler="edm"
outdir="./fid-tmp/cifar10/"$sampler"_"$num_steps
torchrun --standalone --nproc_per_node=$num_gpu generate.py --outdir=$outdir --seeds=0-49999 --subdirs \
    --sampler_type=$sampler --steps=$num_steps --batch=100 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

torchrun --standalone --nproc_per_node=$num_gpu fid.py calc --images=$outdir \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz


sampler="d_edm"
outdir="./fid-tmp/cifar10/"$sampler"_"$num_steps
torchrun --standalone --nproc_per_node=$num_gpu generate.py --outdir=$outdir --seeds=0-49999 --subdirs \
    --sampler_type=$sampler --steps=$num_steps --batch=100 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

torchrun --standalone --nproc_per_node=$num_gpu fid.py calc --images=$outdir \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz


