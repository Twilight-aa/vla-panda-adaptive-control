import json,csv
import run_panda_local as sim
out=sim.ROOT/'results/fair_tuning';selected=json.loads((out/'selected.json').read_text());rows=[]
for std in [0,.001,.005]:
 for seed in [41,42,43]:
  for method in ['CTC','SMC','RBF_SMC']:
   r=sim.run(method,'combined',seed,out,options={**selected[method],'noise_std':std,'save_traces':False,'initial_std':.02})
   rows.append(dict(noise_std=std,**r))
 with (out/'noise_test.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(std,flush=True)
