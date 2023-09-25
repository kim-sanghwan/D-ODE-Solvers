steps="10"
data="cifar10"
scale="10"
#########################################################################
# Sampling with DDIM
#########################################################################

sampleMethod='ddim'
DIS="uniform_torch"
workdir="experiments/"$data"/ddim_"$steps

python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --eta 0 --ni \
					--skip_type=$DIS --sample_type=$sampleMethod --d_ode_scale=$scale --port 12091

sampleMethod='d_ddim'
DIS="uniform_torch"
workdir="experiments/"$data"/d_ddim_"$steps

python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --eta 0 --ni \
					--skip_type=$DIS --sample_type=$sampleMethod --d_ode_scale=$scale --port 12091

#########################################################################
# Sampling with iPNDM
#########################################################################

sampleMethod='ipndm'
workdir="experiments/"$data"/ipndm_"$steps

python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --ni \
				--sample_type=$sampleMethod --d_ode_scale=$scale --port 12362

sampleMethod='d_ipndm'
workdir="experiments/"$data"/d_ipndm_"$steps

python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --ni \
				--sample_type=$sampleMethod --d_ode_scale=$scale --port 12362 	

#########################################################################
# Sampling with DPM-Solver
#########################################################################

sampleMethod='dpmsolver'
type="dpmsolver"
DIS="time_uniform"
order="2"
scale="10"
method="singlestep"
workdir="experiments/"$data"/dpmsolver2_"$steps

python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --eta 0 --ni \
				--skip_type=$DIS --sample_type=$sampleMethod --dpm_solver_order=$order --dpm_solver_method=$method \
				--dpm_solver_type=$type --d_ode_scale=$scale --port 12381


sampleMethod='d_dpmsolver'
type="dpmsolver"
DIS="time_uniform"
order="2"
scale="10"
method="singlestep"
workdir="experiments/"$data"/d_dpmsolver2_"$steps

python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --eta 0 --ni \
				--skip_type=$DIS --sample_type=$sampleMethod --dpm_solver_order=$order --dpm_solver_method=$method \
				--dpm_solver_type=$type --d_ode_scale=$scale --port 12381

#########################################################################
# Sampling with tAB-DEIS
#########################################################################

sampleMethod='tab_deis'
order="2"
scale="10"
workdir="experiments/"$data"/deis2_"$steps

python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --ni \
				--sample_type=$sampleMethod --deis_order=$order --d_ode_scale=$scale --port 12371

sampleMethod='d_tab_deis'
order="2"
scale="10"
workdir="experiments/"$data"/d_deis2_"$steps

python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --ni \
				--sample_type=$sampleMethod --deis_order=$order --d_ode_scale=$scale --port 12371
