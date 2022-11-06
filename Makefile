build:
	python setup.py build_ext --inplace

bin5: scripts/run_experiments.py
	python -m scripts.run_experiments --synth-bin -d 5 -s 156 312 625 1250 2500 5000 10000 -R=-3 -k=4 -r=5 -i=5000 > run_experiments_bin5.log 2>&1

synth: scripts/run_experiments.py
	python -m scripts.run_experiments --synth-bin --synth-mix -d 10 -s 156 312 625 1250 2500 5000 10000 -R=-3 -k=4 -r=5 -i=5000 > run_experiments_synth.log 2>&1

real: scripts/run_experiments.py
	python -m scripts.run_experiments --chemreact --webspam --covtype -d 0 -s 156 312 625 1250 2500 5000 10000 -R=-3 -k=6 -r=5 -i=5000 > run_experiments_real.log 2>&1

test:
	python -m pytest .
