export PYTHONPATH=""
export PYTHONPATH=$PYTHONPATH:/home/groups/kipac/chto/package/Pylians3/library/build/lib.linux-x86_64-cpython-310
mpirun -np 25 /home/groups/risahw/chto/miniconda3/envs/mpi4py-fft/bin/python make_lagfields_project_mpi.py test_glass_highres.yaml
mpirun -np 25 /home/groups/risahw/chto/miniconda3/envs/mpi4py-fft/bin/python make_lagfields_project_mpi.py illustris_300-1_field.yaml
export PYTHONPATH=""
export PYTHONPATH=$PYTHONPATH:/home/groups/kipac/chto/package/Pylians3/library/build/lib.linux-x86_64-3.8
mpirun -np 25 /home/groups/risahw/chto/miniconda3/envs/nbodykit-env/bin/python  measure_basis_project.py illustris_300-1_field.yaml z0
mpirun -np 25 /home/groups/risahw/chto/miniconda3/envs/nbodykit-env/bin/python  measure_basis_project.py test_glass_highres.yaml z0

