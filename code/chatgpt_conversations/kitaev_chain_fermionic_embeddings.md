# Debug Kitaev chain notebook

> Analyse and debug code/fermionic_cases/kitaev_chain/quspin_kitaev_chain_defect_operators.ipynb.

<details><summary>10 previous messages</summary>

> I’ll inspect the notebook, trace its calculations and dependencies, and fix any bugs I can reproduce.
>
> <details><summary>Ran 6 commands, searched the web</summary>
>
> - Ran `pwd; rg --files -g AGENTS.md -g '*quspin_kitaev_chain_defect_operators*' -g '*kitaev*'`
>
> - Ran `git status --short; for p in /home /home/kcooney /home/kcooney/Desktop /home/kcooney/Desktop/repos /home/kcooney/Desktop/repos/spt_numerical_classificiation /home/kcooney/Desktop/repos/spt_numerical_classificiation/code /home/kcooney/Desktop/repos/spt_numerical_classificiation/code/time_reversal_symmetries /home/kcooney/Desktop/repos/spt_numerical_classificiation/code/time_reversal_symmetries/disentangling_projectors; do if [ -f "$p/AGENTS.md" ]; then cat "$p/AGENTS.md"; fi; done`
>
> - Ran `rg --files ../../fermionic_cases -g AGENTS.md -g '*kitaev*'; python - <<'PY'
> import json
> p='../../fermionic_cases/kitaev_chain/quspin_kitaev_chain_defect_operators.ipynb'
> n=json.load(open(p))
> print(n['metadata'])
> for i,c in enumerate(n['cells']):
>  print('\n### CELL',i,c['cell_type']);print(''.join(c['source']))
>  for o in c.get('outputs',[]):
>   if o.get('output_type')=='error': print('ERROR',o)
>   elif 'text' in o: print('OUTPUT',''.join(o['text'])[:1500])
> PY`
>
> - Ran `ls ../../num_spt_venv_p314/bin/python*; rg --files ../../fermionic_cases -g AGENTS.md; ls ../../make_plots/data/kitaev_chain_defect_operators; ../../num_spt_venv_p314/bin/python - <<'PY'
> import inspect,quspin,quimb
> from quspin.basis import spinless_fermion_basis_1d
> print(quspin.__version__, quimb.__version__)
> print(inspect.getsource(spinless_fermion_basis_1d.partial_trace))
> PY`
>
> - Ran `NUMBA_CACHE_DIR=/tmp/kitaev-numba-cache /home/kcooney/Desktop/repos/spt_numerical_classificiation/code/num_spt_venv_p314/bin/python - <<'PY'
> import inspect,quspin,quimb
> from quspin.basis import spinless_fermion_basis_1d
> print(quspin.__version__,quimb.__version__)
> print(inspect.getsource(spinless_fermion_basis_1d.partial_trace))
> print(inspect.getsource(spinless_fermion_basis_1d._partial_trace))
> PY`
>
> - Ran `rg --files ../../fermionic_cases/kitaev_chain; python - <<'PY'
> import json
> n=json.load(open('../../fermionic_cases/kitaev_chain/kitaev_chain_two_site_defect_operators.ipynb'))
> for i,c in enumerate(n['cells']):
>  s=''.join(c['source'])
>  if any(x in s for x in ['partial_trace','symmetry_op','extract_n1','fp_matrix']):print(i,s[:6000])
> PY`
>
> Searched the web for `site.quspin.github.io spinless_fermion_basis_1d partial_trace subsys_ordering`
>
> </details>
>
> I found a clean-run failure (`all_results` is used before it is defined), and the final “Old code” cell silently replaces `extract_n1` with a different implementation. The correlation-length calculation also takes logs of zero or signed values. I’m checking the fermionic trace conventions and optimizer against numerical examples before changing those parts.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `cat > /tmp/probe_kitaev.py <<'PY'
> import json,numpy as np
> n=json.load(open('/home/kcooney/Desktop/repos/spt_numerical_classificiation/code/fermionic_cases/kitaev_chain/quspin_kitaev_chain_defect_operators.ipynb'))
> g={}
> for i in [2,3,10,18,49,50,51,52,54,57]:exec(''.join(n['cells'][i]['source']),g)
> for t in [0,.2,.5,1]:
>  h=g['get_kitaev_chain_hamiltonian'](t,16)
>  e,v=h.eigsh(k=1,which='SA',tol=1e-11,v0=np.random.default_rng(1).normal(size=h.Ns))
>  for k in range(2):
>   r=g['extract_n1'](v[:,0]);print(t,e, r[:2],r[2][:3],r[2][-1],flush=True)
> PY
> NUMBA_CACHE_DIR=/tmp/kitaev-numba-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/kcooney/Desktop/repos/spt_numerical_classificiation/code/num_spt_venv_p314/bin/python /tmp/probe_kitaev.py`
> - Ran `cat > /tmp/edit_kitaev.py <<'PY'
> import json
> from pathlib import Path
> src=Path('/home/kcooney/Desktop/repos/spt_numerical_classificiation/code/fermionic_cases/kitaev_chain/quspin_kitaev_chain_defect_operators.ipynb')
> n=json.loads(src.read_text())
> def setcell(i,s):n['cells'][i]['source']=s.strip('\n').splitlines(keepends=True)
> setcell(4,'''from pathlib import Path
>
> # One source of truth for the system size and random initialization.
> num_sites = 16
> random_seed = 0''')
> setcell(9,'## Find ground states')
> s=''.join(n['cells'][10]['source']).replace('    dynamic = []\n','').replace('dtype=np.complex128','dtype=np.float64')
> setcell(10,s)
> setcell(12,'''# Build one sparse Hamiltonian at a time; retaining all 201 is unnecessary.
> assert num_sites >= 14, "The defect regions below require at least 14 sites."''')
> setcell(13,'''energies = []
> groundstates = []
> eigenstate_residuals = []
> rng = np.random.default_rng(random_seed)
>
> for t in tqdm(parameters):
>     h = get_kitaev_chain_hamiltonian(t, num_sites)
>     # A fresh generic start can reach either fermion-parity sector.
>     e, psi = h.eigsh(k=1, which='SA', tol=1e-11,
>                      v0=rng.normal(size=h.Ns))
>     energies.append(e[0])
>     groundstates.append(psi[:, 0])
>     eigenstate_residuals.append(np.linalg.norm(h.dot(psi[:, 0]) - e[0]*psi[:, 0]))
>     del h
>
> assert max(eigenstate_residuals) < 1e-8''')
> setcell(18,'middle_site = num_sites//2')
> setcell(21,'''n_middle_site_exps = np.real_if_close([
>     n_middle_site_op.expt_value(psi)
>     for psi in groundstates
> ])''')
> setcell(25,'n_site_exps = np.real_if_close(np.array(n_site_exps))')
> setcell(27,'''plt.plot(parameters, n_site_exps.T);
> plt.xlabel("t");
> plt.ylabel("Site occupation");''')
> setcell(29,'''n_two_site_exps = np.real_if_close(np.array([
>     [op.expt_value(psi) for psi in groundstates]
>     for op in n_two_site_ops
> ]))''')
> setcell(33,'''n_correlations = np.real_if_close(
>     n_two_site_exps - n_site_exps[0]*n_site_exps[:num_sites//2]
> )
> assert not np.iscomplexobj(n_correlations), "Unexpected imaginary density correlation."
>
> # Signed/vanishing correlations do not have a real logarithm.
> # Mask values at the numerical noise floor instead of manufacturing zeros.
> correlation_atol = 1e-12
>
> def log_correlation_magnitude(values, atol=correlation_atol):
>     magnitude = np.abs(values)
>     out = np.full(magnitude.shape, np.nan, dtype=float)
>     np.log(magnitude, out=out, where=magnitude > atol)
>     return out
>
> def effective_correlation_length(correlations, r1, r2, atol=correlation_atol):
>     """Two-point decay estimate; NaN when the magnitude does not resolve a decay.
>
>     Assumes |C(r)| ~ A exp(-r/xi). Finite-ring and non-exponential
>     corrections mean this is an effective estimate, not an asymptotic fit.
>     """
>     if not 0 <= r1 < r2 < len(correlations):
>         raise ValueError("Require 0 <= r1 < r2 < number of separations.")
>     a, b = np.abs(correlations[r1]), np.abs(correlations[r2])
>     valid = (a > atol) & (b > atol) & (b < a)
>     log_ratio = log_correlation_magnitude(b, atol) - log_correlation_magnitude(a, atol)
>     out = np.full(np.shape(a), np.nan, dtype=float)
>     np.divide(-(r2-r1), log_ratio, out=out, where=valid)
>     return out''')
> setcell(36,'plt.plot(log_correlation_magnitude(n_correlations[:, 5]));')
> setcell(37,'effective_correlation_length(n_correlations[:, 10], 0, 5)')
> setcell(38,'plt.plot(log_correlation_magnitude(n_correlations));')
> setcell(40,'plt.plot(log_correlation_magnitude(n_correlations[:, 60:80]));')
> setcell(41,'corr_lengths = effective_correlation_length(n_correlations, 1, 3)')
> setcell(42,'np.count_nonzero(np.isfinite(corr_lengths))')
> setcell(43,'# Unresolved estimates stay NaN; do not replace them with zero or huge finite values.')
> setcell(47,'''fig, ax = plt.subplots(figsize=(6,4))
> ax.plot(parameters, corr_lengths)
> ax.axvline(1/3, color='red')
> ax.set(xlabel="t", ylabel="Effective density correlation length")
> ax.grid()''')
> setcell(49,'''symmetry_sites = list(range(5, 11))
> left_defect_sites = [3, 4]
> right_defect_sites = [11, 12]
> all_defect_sites = left_defect_sites + right_defect_sites
> # QuSpin sorts sub_sys_A by default. Make that ordering explicit.
> retained_sites = sorted(all_defect_sites + symmetry_sites)
> rdm_symmetry_sites = [retained_sites.index(i) for i in symmetry_sites]
> assert retained_sites == left_defect_sites + symmetry_sites + right_defect_sites
>
> num_symmetry_sites = len(symmetry_sites)
> num_defect_sites = len(all_defect_sites)
> num_symmetry_and_defect_sites = len(retained_sites)''')
> setcell(51,'''# z = n - 1/2, so physical parity prod(1 - 2*n) = prod(-2*z).
> symmetry_op = quantum_operator(
>     {"fp": [["z"*num_symmetry_sites,
>               [[(-2)**num_symmetry_sites, *rdm_symmetry_sites]]]]},
>     basis=symmetry_and_defect_basis,
>     dtype=np.complex128
> )
>
> defect_basis = spinless_fermion_basis_1d(num_defect_sites, Nf=None)
> fp_matrix = np.diag([1, -1, -1, 1])''')
> setcell(54,'''def get_defect_operators(edm, num_iters=20, seed=random_seed):
>     """Alternate unitary polar updates to maximize |Tr((L tensor R) E)|.
>
>     This is a local alternating optimization, not a global optimality proof.
>     Explicit index order makes updates independent of tensor storage order.
>     """
>     if num_iters < 1:
>         raise ValueError("num_iters must be positive")
>     scores = []
>     rng = np.random.default_rng(seed)
>     dl, dr = edm.ind_size('kl1'), edm.ind_size('kr2')
>     if edm.ind_size('bl1') != dl or edm.ind_size('br2') != dr:
>         raise ValueError("Each defect environment must be square")
>     initial = rng.normal(size=(dl, dl)) + 1j*rng.normal(size=(dl, dl))
>     # The initial defect must also be unitary for scores to be bounded by 1.
>     left_defect_op = qtn.Tensor(np.linalg.qr(initial)[0], inds=['kl1', 'bl1'])
>
>     for _ in range(num_iters):
>         right_edm = (edm & left_defect_op.reindex(
>             {'kl1': 'bl1', 'bl1': 'kl1'})).contract()
>         data = right_edm.transpose('kr2', 'br2').data
>         sol, score = get_defect_operator_from_one_sided_environment_matrix(data)
>         right_defect_op = qtn.Tensor(sol, inds=['kr2', 'br2'])
>         scores.append(score)
>
>         left_edm = (edm & right_defect_op.reindex(
>             {'kr2': 'br2', 'br2': 'kr2'})).contract()
>         data = left_edm.transpose('kl1', 'bl1').data
>         sol, score = get_defect_operator_from_one_sided_environment_matrix(data)
>         left_defect_op = qtn.Tensor(sol, inds=['kl1', 'bl1'])
>         scores.append(score)
>
>     return (left_defect_op, right_defect_op), scores''')
> s=''.join(n['cells'][57]['source']).replace('sub_sys_A=all_defect_sites+symmetry_sites,','sub_sys_A=retained_sites,\n        subsys_ordering=True,').replace('    edm = edm[0]','''    # QuSpin 1.0.1 returns a singleton batch axis for a mixed input.
>     edm = np.asarray(edm).reshape(2**num_defect_sites, 2**num_defect_sites)''').replace('    # Hard code warning.','    # Each boundary contains two sites.').replace('    reshaped_edm = edm.reshape(4,4,4,4)','    dl, dr = 2**len(left_defect_sites), 2**len(right_defect_sites)\n    reshaped_edm = edm.reshape(dl, dr, dl, dr)')
> setcell(57,s)
> setcell(66,'# Assemble results before inspecting their shape.')
> s=''.join(n['cells'][67]['source'])+'\nassert all_results.shape == (len(parameters), 3)\nall_results.shape'
> setcell(67,s)
> setcell(68,'''# Locate code/ from either the notebook directory or another directory in the repo.
> code_dir = next((p for p in (Path.cwd(), *Path.cwd().parents)
>                  if (p / "fermionic_cases" / "kitaev_chain").is_dir()), None)
> if code_dir is None:
>     raise FileNotFoundError("Run this notebook from within the repository's code/ directory.")
> results_path = code_dir / "make_plots/data/kitaev_chain_defect_operators/results.pkl"
> results_path.parent.mkdir(parents=True, exist_ok=True)
> with results_path.open('wb') as f:
>     pickle.dump(all_results, f)''')
> setcell(69,'''# Interpretation and numerical limits
>
> The two columns of parity indices should approach +1 in the trivial limit
> (t = 0) and -1 in the topological limit (t = 1); the third column is the
> optimized defect overlap. The Hamiltonian uses physical periodic fermionic
> boundary conditions. The bulk gap closes at t = 1/3; a finite system can have
> degenerate ground states there, so a single eigensolver vector need not give
> a unique index at the transition.
>
> The density correlation length is a two-separation effective estimate. NaN
> means the decay was unresolved or not decreasing; it does not mean zero
> correlation length. Alternating unitary optimization is local and can depend
> on initialization. Its seed is fixed here for reproducibility.
>
> QuSpin's fermionic partial trace includes reordering signs; replacing it with
> an ordinary tensor trace changes the disconnected-subsystem convention.
> See [QuSpin partial-trace documentation](https://quspin.github.io/QuSpin/generated/quspin.basis.spinless_fermion_basis_1d.html).
> The parity-string normalization now uses prod(1-2n); the former overall minus
> sign changed the environment phase, not the optimized overlap or parity index.''')
> setcell(70,'# Legacy Schmidt-factor approximation (not used in the sweep)')
> s=''.join(n['cells'][71]['source']).replace('def extract_n1(psi):','def extract_n1_schmidt_legacy(psi):').replace('    edm = edm[0]','    edm = np.asarray(edm).reshape(2**num_defect_sites, 2**num_defect_sites)')
> pos=s.index('    right_index =')
> s=s[:pos]+s[pos:].replace('@ rho_l','@ rho_r')
> setcell(71,s)
> for c in n['cells']:
>  if c['cell_type']=='code':c['execution_count']=None;c['outputs']=[]
> Path('/tmp/quspin_kitaev_chain_defect_operators.ipynb').write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n')
> PY
> python /tmp/edit_kitaev.py`
>
> </details>
>
> The endpoint checks give the expected parity indices: +1 at \(t=0\) and −1 at \(t=1\). I also reproduced optimizer scores above 1 on the first update because its initial matrix was not unitary.
>
> I’ve prepared fixes for that initialization, the execution-order bugs, and invalid correlation-length estimates. I’m now testing the full sweep and checking the Hamiltonian against an independent construction.
>
> <details><summary>Ran 8 commands</summary>
>
> - Ran `cat > /tmp/test_kitaev.py <<'PY'
> import json, time, warnings
> import numpy as np
> import matplotlib
> matplotlib.use('Agg')
> import matplotlib.pyplot as plt
> n=json.load(open('/tmp/quspin_kitaev_chain_defect_operators.ipynb'))
> g={}
> start=time.time()
> for i,c in enumerate(n['cells']):
>  if c['cell_type']!='code':continue
>  s=''.join(c['source'])
>  # Exercise the save cell without overwriting the user's previous results.
>  if i==68:s=s.replace('code_dir / "make_plots/data/kitaev_chain_defect_operators/results.pkl"','Path("/tmp/kitaev-debug-results.pkl")')
>  with warnings.catch_warnings():
>   warnings.simplefilter('error', RuntimeWarning)
>   warnings.simplefilter('error', np.exceptions.ComplexWarning)
>   exec(compile(s,f'cell_{i}','exec'),g)
>  plt.close('all')
>  if i in [13,47,58,68,71]:print('Completed cell',i,'elapsed',round(time.time()-start,1),flush=True)
> # Check clean-run function is not overwritten by the legacy cell.
> assert g['extract_n1'].__code__.co_filename=='cell_57'
> assert np.allclose(g['indices_array'][0],1,atol=1e-7)
> assert np.allclose(g['indices_array'][-1],-1,atol=1e-7)
> for _,_,scores in g['results']:
>  assert np.all(np.asarray(scores)<=1+1e-9)
>  assert np.min(np.diff(scores))>-1e-9
> assert not np.isinf(g['corr_lengths']).any()
> # Independent occupation-basis Hamiltonian; site 0 is the most significant bit.
> L=4;dim=2**L
> ann=[]
> for i in range(L):
>  a=np.zeros((dim,dim))
>  for col in range(dim):
>   state=dim-1-col
>   if state & (1 << (L-1-i)):
>    target=state ^ (1 << (L-1-i))
>    sign=(-1)**sum((state >> (L-1-j))&1 for j in range(i))
>    a[dim-1-target,col]=sign
>  ann.append(a)
> for t in [0,.2,1/3,.5,1]:
>  h=g['get_kitaev_chain_hamiltonian'](t,L).toarray()
>  ref=np.zeros((dim,dim))
>  for i in range(L):
>   j=(i+1)%L;a,b=ann[i],ann[j]
>   ref += -t*(a.T@b+b.T@a)+t*(a@b+b.T@a.T)+(1-t)*(.5*np.eye(dim)-a.T@a)
>  assert np.allclose(h,h.T)
>  assert np.allclose(h,ref)
> # Complex environments and deliberately permuted tensor axes.
> rng=np.random.default_rng(2)
> a=rng.normal(size=(16,16))+1j*rng.normal(size=(16,16))
> e=a+a.conj().T;e/=np.linalg.svd(e,compute_uv=False).sum()
> tensor=g['qtn'].Tensor(e.reshape(4,4,4,4),inds=['kl1','kr2','bl1','br2'])
> (lt,rt),scores=g['get_defect_operators'](tensor)
> l,r=lt.data,rt.data
> assert np.allclose(l.conj().T@l,np.eye(4))
> assert np.allclose(r.conj().T@r,np.eye(4))
> assert np.allclose(np.trace(np.kron(l,r)@e),scores[-1])
> (_, _),reordered=g['get_defect_operators'](tensor.transpose('br2','kl1','bl1','kr2'))
> assert np.allclose(scores,reordered)
> assert np.min(np.diff(scores))>-1e-12
> c=np.exp(-np.arange(8)/2.5)
> assert np.allclose(g['effective_correlation_length'](c,1,3),2.5)
> assert np.isnan(g['effective_correlation_length'](np.zeros(8),1,3))
> assert np.isnan(g['effective_correlation_length'](np.ones(8),1,3))
> assert np.allclose(g['effective_correlation_length'](-c,1,3),2.5)
> print('PASS: full 201-point/16-site clean run; independent Hamiltonian; endpoint indices; bounded monotone scores; tensor-order invariance; correlation estimates.',flush=True)
> print('Max eigensolver residual:',max(g['eigenstate_residuals']))
> print('Endpoint results:',g['all_results'][[0,-1]])
> print('Saved results shape:',g['all_results'].shape)
> PY
> NUMBA_CACHE_DIR=/tmp/kitaev-numba-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/kitaev-mpl /home/kcooney/Desktop/repos/spt_numerical_classificiation/code/num_spt_venv_p314/bin/python /tmp/test_kitaev.py > /tmp/kitaev-validation.log 2>&1`
> - Ran `python - <<'PY'
> import json,difflib
> old=json.load(open('../../fermionic_cases/kitaev_chain/quspin_kitaev_chain_defect_operators.ipynb'))
> new=json.load(open('/tmp/quspin_kitaev_chain_defect_operators.ipynb'))
> for i,(a,b) in enumerate(zip(old['cells'],new['cells'])):
>  if a['source']!=b['source']:
>   print('CELL',i)
>   print(''.join(difflib.unified_diff(a['source'],b['source'])),end='\n')
> PY`
> - Ran `tail -c 2500 /tmp/kitaev-validation.log`
> - Ran `git diff --numstat -- ../../fermionic_cases/kitaev_chain/quspin_kitaev_chain_defect_operators.ipynb; ls -lh ../../fermionic_cases/kitaev_chain/quspin_kitaev_chain_defect_operators.ipynb /tmp/quspin_kitaev_chain_defect_operators.ipynb`
> - Ran `tail -c 1200 /tmp/kitaev-validation.log`
> - Ran `/home/kcooney/Desktop/repos/spt_numerical_classificiation/code/num_spt_venv_p314/bin/python - <<'PY'
> import nbformat
> n=nbformat.read('/tmp/quspin_kitaev_chain_defect_operators.ipynb',as_version=4)
> nbformat.validate(n)
> print('Notebook schema valid; cells:',len(n.cells))
> PY`
> - Ran `python - <<'PY'
> import nbformat
> n=nbformat.read('/tmp/quspin_kitaev_chain_defect_operators.ipynb',as_version=4)
> nbformat.validate(n)
> print('Notebook schema valid; cells:',len(n.cells))
> PY`
> - Ran `tail -c 500 /tmp/kitaev-validation.log`
>
> </details>
>
> The full sweep is progressing without numerical warnings. I also corrected the legacy function’s right-hand index, which used the left reduced density matrix. The revised notebook builds one Hamiltonian at a time and documents the limitations of the correlation-length estimate and local optimizer.
>
> <details><summary>Ran 5 commands</summary>
>
> - Ran `tail -c 600 /tmp/kitaev-validation.log`
> - Ran `tail -c 1500 /tmp/kitaev-validation.log`
> - Ran `tail -c 1600 /tmp/kitaev-validation.log`
> - Ran `cat > /tmp/check_kitaev_convention.py <<'PY'
> import json,numpy as np
> from quspin.operators import hamiltonian
> from quspin.basis import spinless_fermion_basis_1d
> L=4;dim=2**L;b=spinless_fermion_basis_1d(L)
> for i in range(2):
>  a=np.zeros((dim,dim))
>  for col in range(dim):
>   state=dim-1-col
>   if state & (1 << (L-1-i)):
>    target=state^(1<<(L-1-i));sign=(-1)**sum((state>>(L-1-j))&1 for j in range(i))
>    a[dim-1-target,col]=sign
>  q=hamiltonian([['-',[[1,i]]]],[],basis=b,dtype=np.float64,check_herm=False,check_symm=False,check_pcon=False).toarray()
>  print(i,'quspin',q[np.nonzero(q)],'reference',a[np.nonzero(a)])
>  print('same positions',np.array_equal(q!=0,a!=0))
> print(b.states)
> PY
> NUMBA_CACHE_DIR=/tmp/kitaev-numba-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/kcooney/Desktop/repos/spt_numerical_classificiation/code/num_spt_venv_p314/bin/python /tmp/check_kitaev_convention.py`
> - Ran `python - <<'PY'
> p='/tmp/test_kitaev.py'
> s=open(p).read().replace('for j in range(i))','for j in range(i+1,L))')
> s=s.replace('# Independent occupation-basis Hamiltonian; site 0 is the most significant bit.','# Independent occupation-basis Hamiltonian; site 0 is the most significant bit.\n# QuSpin counts occupied modes to the RIGHT of the acted-on site.')
> open(p,'w').write(s)
> part=s[s.index('# Independent occupation'):s.index("print('PASS:")]
> init="import json, numpy as np\nn=json.load(open('/tmp/quspin_kitaev_chain_defect_operators.ipynb'))\ng={}\nfor i in [2,3,4,10,33,52,54]:\n if i==33:\n  text=''.join(n['cells'][i]['source']); text=text[text.index('correlation_atol ='):]\n else:text=''.join(n['cells'][i]['source'])\n exec(text,g)\n"
> open('/tmp/kitaev_independent_tests.py','w').write(init+part+"\nprint('PASS: independent Hamiltonian, unitary constraints, objective equality, tensor-order invariance, correlation estimates.')\n")
> PY
> NUMBA_CACHE_DIR=/tmp/kitaev-numba-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/kcooney/Desktop/repos/spt_numerical_classificiation/code/num_spt_venv_p314/bin/python /tmp/kitaev_independent_tests.py`
>
> </details>
>
> All 201 parameter points ran successfully on 16 sites, with the expected endpoint indices and bounded, nondecreasing optimizer scores. The independent Hamiltonian and tensor-order checks also pass.
>
> The tested notebook is ready to install. Its location is outside this chat’s writable workspace, so the filesystem sandbox requires approval for that final write.
>
> <details><summary>Ran a command</summary>
>
> - Ran `/bin/bash -lc 'cp /tmp/quspin_kitaev_chain_defect_operators.ipynb /home/kcooney/Desktop/repos/spt_numerical_classificiation/code/fermionic_cases/kitaev_chain/quspin_kitaev_chain_defect_operators.ipynb'`
>
> </details>
</details>

Prepared a [corrected notebook](/tmp/quspin_kitaev_chain_defect_operators.ipynb), but **the original remains unchanged because write approval was declined**.

Fixes include:

- `all_results` referenced before assignment.
- Legacy code overwriting `extract_n1` and using `rho_l` for the right index.
- Nonunitary initialization producing optimizer scores above 1.
- Invalid correlation logarithms, inconsistent variable names, and incorrect distance factor.
- Excessive memory use from retaining all 201 Hamiltonians.

Validation passed: the complete 201-point, 16-site calculation, independent Hamiltonian checks, unitary and tensor-order checks, and expected endpoint indices (+1 and −1).

The notebook also documents remaining limitations: correlation lengths are effective estimates, and alternating optimization does not guarantee a global optimum.

> Please save the corrected notebook as a copy (prepend chatgpt_ to the filename) in the same directory as the original notebook.

<details><summary>2 previous messages</summary>

> I’ll save the corrected notebook beside the original with the `chatgpt_` prefix. That directory is outside this chat’s writable workspace, so the save requires filesystem approval.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `cp -n /tmp/quspin_kitaev_chain_defect_operators.ipynb /home/kcooney/Desktop/repos/spt_numerical_classificiation/code/fermionic_cases/kitaev_chain/chatgpt_quspin_kitaev_chain_defect_operators.ipynb`
> - Ran `cmp /tmp/quspin_kitaev_chain_defect_operators.ipynb /home/kcooney/Desktop/repos/spt_numerical_classificiation/code/fermionic_cases/kitaev_chain/chatgpt_quspin_kitaev_chain_defect_operators.ipynb`
>
> </details>
</details>

Saved as [chatgpt_quspin_kitaev_chain_defect_operators.ipynb](~/Desktop/repos/spt_numerical_classificiation/code/fermionic_cases/kitaev_chain/chatgpt_quspin_kitaev_chain_defect_operators.ipynb). The original notebook is unchanged.

> Can you justify theoretically what the notebook is trying to do?

<details><summary>2 previous messages</summary>

> I’ll connect the Hamiltonian, parity-string construction, and SVD updates to the phase diagnostic, and distinguish the theoretical invariant from its finite-size numerical estimate.
>
> <details><summary>Searched the web</summary>
>
> Searched the web for `site.arxiv.org Turner Pollmann Berg topological phases one dimensional fermions entanglement 2011 parity ...`
>
> Searched the web for `https://arxiv.org/pdf/1008.4346`
>
> </details>
</details>

Yes. **The notebook is trying to detect the Kitaev chain’s topological phase by finding the boundary operators associated with a finite fermion-parity string, then testing whether those operators are even or odd under fermion parity.**

That has a theoretical foundation in symmetry fractionalization. The distinction is between the ideal, quantized invariant and the finite-region, variational estimate computed here.

**1. The Hamiltonian crosses the expected topological transition.**

The code implements
\[
H(t)=\sum_j\left[
-t(c_j^\dagger c_{j+1}+c_{j+1}^\dagger c_j)
+t(c_jc_{j+1}+c_{j+1}^\dagger c_j^\dagger)
-(1-t)\left(n_j-\frac12\right)
\right],
\]
with periodic fermionic boundary conditions.

Thus the hopping and pairing magnitudes are \(t\), and the chemical potential is \(\mu=1-t\). Its bulk quasiparticle dispersion is
\[
\varepsilon(k)=
\sqrt{(\mu+2t\cos k)^2+4t^2\sin^2k}.
\]
The gap closes when \(\mu=2t\), giving
\[
t_c=\frac13.
\]

The corresponding free-fermion \(\mathbb Z_2\) invariant is
\[
\mathcal M
=\operatorname{sgn}\!\left[(\mu+2t)(\mu-2t)\right]
=\operatorname{sgn}\!\left[(1+t)(1-3t)\right].
\]
Consequently, the expected phases are trivial for \(t<1/3\) and topological for \(t>1/3\). The latter supports unpaired Majorana boundary modes in an open chain. [Kitaev’s original construction](https://arxiv.org/abs/cond-mat/0010440)

**2. A partial symmetry operation produces effects at its two boundaries.**

The conserved symmetry is total fermion parity,
\[
P=\prod_j(1-2n_j).
\]
The notebook applies this symmetry only within an interval \(I\):
\[
P_I=\prod_{j\in I}(1-2n_j).
\]

Conjugating the Hamiltonian by \(P_I\) leaves terms entirely inside or outside \(I\) unchanged. Only terms crossing its two boundaries change. This is why it makes sense to seek *defect operators* near those boundaries.

For a gapped state, symmetry acts on the relevant entanglement degrees of freedom through boundary factors. Their fermionic character distinguishes the phases: the factors can be even in the trivial phase, whereas the Kitaev phase has odd factors at both ends. This is the parity-fractionalization component of the entanglement classification of one-dimensional fermions. [Turner, Pollmann, and Berg](https://arxiv.org/abs/1008.4346)

Your geometry is
\[
\underbrace{3,4}_{L}
\quad
\underbrace{5,\ldots,10}_{I}
\quad
\underbrace{11,12}_{R}.
\]
The numerical question is whether operators supported on \(L\) and \(R\) can compensate for applying \(P_I\).

**3. The environment matrix converts that question into an optimization problem.**

The notebook first forms
\[
\rho_{LIR}=\operatorname{Tr}_{\mathrm{outside}}|\psi\rangle\langle\psi|,
\]
then constructs
\[
E_{LR}=\operatorname{Tr}_{I}\!\left(P_I\rho_{LIR}\right),
\]
using QuSpin’s fermionic ordering conventions.

Its objective is
\[
s_{\max}
=\max_{D_L,D_R\ \mathrm{unitary}}
\left|\operatorname{Tr}\!\left[(D_L\otimes D_R)E_{LR}\right]\right|.
\]

This is a boundary-corrected parity-string overlap. A value near one means that the optimized correction approximately restores the original state, up to phase.

There is a necessary fermionic qualification: **the Kronecker product is written in the ordered reduced Fock-space representation.** Interpreting its factors as physical odd operators on separated regions requires the corresponding parity/Jordan–Wigner factors. Those factors can be absorbed into the boundary matrices without changing their even/odd character. An ordinary spin partial trace would not generally represent the same calculation.

Also, a large score alone is not the topological invariant: both fixed-point phases can achieve score one. The distinguishing information is the parity of the boundary factors.

**4. The SVD update is an exact solution of each one-sided optimization.**

Fix \(D_L\) and contract it into the environment, obtaining a matrix \(M_R\). The remaining problem is
\[
\max_{D_R^\dagger D_R=I}
\left|\operatorname{Tr}(D_RM_R)\right|.
\]
If
\[
M_R=U\Sigma V^\dagger,
\]
then
\[
D_R=VU^\dagger
\]
achieves
\[
\operatorname{Tr}(D_RM_R)=\operatorname{Tr}\Sigma=\|M_R\|_*.
\]

That is precisely the notebook’s update
```python
sol = (U @ Vh).conj().T
```
and its score `sum(S)`.

Alternating left and right updates therefore gives nondecreasing scores when initialized with unitary matrices. Each update is optimal with the other factor fixed; the joint problem remains nonconvex, so this is not a guarantee of a global optimum.

**5. The reported index tests whether a defect operator is fermion-even or fermion-odd.**

For either boundary region \(A\), the code evaluates
\[
\nu_A=
\operatorname{Tr}\!\left(
\rho_A D_A^\dagger P_A D_A P_A
\right).
\]

If \(D_A\) is even,
\[
P_AD_AP_A=D_A,
\qquad \nu_A=+1.
\]
If it is odd,
\[
P_AD_AP_A=-D_A,
\qquad \nu_A=-1.
\]

There is also a useful quantitative interpretation:
\[
\operatorname{Tr}\!\left(\rho_A[D_A,P_A]^\dagger[D_A,P_A]\right)
=2(1-\operatorname{Re}\nu_A),
\]
and
\[
\operatorname{Tr}\!\left(\rho_A\{D_A,P_A\}^\dagger\{D_A,P_A\}\right)
=2(1+\operatorname{Re}\nu_A).
\]
Thus proximity to \(+1\) or \(-1\) directly measures approximate commutation or anticommutation **on the state’s occupied reduced-density-matrix support**.

This explains the notebook’s intended identification:
\[
\boxed{\nu_L\simeq\nu_R\simeq+1\;\text{trivial},
\qquad
\nu_L\simeq\nu_R\simeq-1\;\text{topological}.}
\]

**6. The two fixed points make the interpretation explicit.**

At \(t=0\), the ground state is the fully occupied product state. Applying \(P_I\) changes only an overall sign. Identity boundary corrections suffice, giving
\[
s=1,\qquad \nu_L=\nu_R=+1.
\]

At \(t=1\), introduce Majoranas
\[
a_j=c_j+c_j^\dagger,\qquad
b_j=-i(c_j-c_j^\dagger).
\]
The Hamiltonian becomes
\[
H(1)=i\sum_j b_ja_{j+1}.
\]
Its ground state consists of intersite Majorana dimers.

Since \(1-2n_j=-ia_jb_j\), the interval parity string reduces, on this dimer state, to endpoint Majoranas multiplied by fixed internal bond eigenvalues. The crossing dimers allow those endpoint actions to be represented on the adjacent exterior sites. The required boundary corrections are therefore odd Majorana operators:
\[
s=1,\qquad \nu_L=\nu_R=-1.
\]

This gives a direct theoretical explanation of the endpoint results we checked numerically.

The main limitation is that **two-site defect regions and a 16-site ring do not establish asymptotic quantization**. Near the transition, the growing localization and correlation lengths make that approximation particularly weak. Unrestricted optimized matrices can mix even and odd components, and competing local optima can further blur the result. A stronger validation would enlarge the defect regions and system size, compare random initializations, and optimize separately within even and odd boundary-operator sectors.

Finally, the density-correlation calculation is supporting evidence about length scales. Its two-distance estimate is not itself a topological diagnostic, nor is the density correlation length guaranteed to equal the boundary-mode localization length. The notebook’s central theoretical content is the **fermion parity of the optimized symmetry-defect factors**.

> "There is a necessary fermionic qualification: **the Kronecker product is written in the ordered reduced Fock-space representation.** Interpreting its factors as physical odd operators on separated regions requires the corresponding parity/Jordan–Wigner factors. Those factors can be absorbed into the boundary matrices without changing their even/odd character. An ordinary spin partial trace would not generally represent the same calculation."
>
> Could you expand on this paragraph.

<details><summary>2 previous messages</summary>

> I’ll work through the sign conventions explicitly. There’s also a qualification to my earlier statement: absorbing the parity factors into a boundary matrix is straightforward when the operators have definite parity, but needs more care for the unrestricted matrices optimized by the notebook.
>
> <details><summary>Ran 2 commands, searched the web</summary>
>
> - Ran `NUMBA_CACHE_DIR=/tmp/kitaev-numba-cache /home/kcooney/Desktop/repos/spt_numerical_classificiation/code/num_spt_venv_p314/bin/python - <<'PY'
> import inspect
> from quspin.basis import spinless_fermion_basis_1d
> print(inspect.getsource(spinless_fermion_basis_1d._ptrace_signs))
> PY`
>
> Searched the web for `site.quspin.github.io fermion partial_trace fermionic signs ordering ...`
>
> - Ran `NUMBA_CACHE_DIR=/tmp/kitaev-numba-cache OPENBLAS_NUM_THREADS=1 /home/kcooney/Desktop/repos/spt_numerical_classificiation/code/num_spt_venv_p314/bin/python - <<'PY'
> import numpy as np
> from quspin.basis import spinless_fermion_basis_1d
> b=spinless_fermion_basis_1d(3)
> rng=np.random.default_rng(7)
> v=rng.normal(size=8)+1j*rng.normal(size=8)
> v[[i for i,s in enumerate(b.states) if int(s).bit_count()%2]]=0
> v/=np.linalg.norm(v);rho=np.outer(v,v.conj())
> P=np.diag([-1.,1.]);X=np.array([[0,1],[1,0]]);I=np.eye(2)
> PI=np.kron(np.kron(I,P),I)
> E=np.asarray(b.partial_trace(PI@rho,sub_sys_A=[1],return_rdm='B')).reshape(4,4)
> for DL,p in [(P,0),(X,1)]:
>  DR=X if p else P
>  q=np.trace(np.kron(DL,DR)@E)
>  ref=np.trace(np.kron(np.kron(DL,np.linalg.matrix_power(P,1+p)),DR)@rho)
>  assert np.allclose(q,ref)
>  print('parity',p,'QuSpin',q,'explicit JW',ref)
> v=np.zeros(8);v[b.index('110')]=1/np.sqrt(2);v[b.index('011')]=1/np.sqrt(2)
> f=np.asarray(b.partial_trace(v,sub_sys_A=[1],return_rdm='B')).reshape(4,4)
> s=np.einsum('lir,L i R->lrLR',np.outer(v,v).reshape(2,2,2,2,2,2)) if False else np.einsum('abcdbf->acdf',np.outer(v,v).reshape(2,2,2,2,2,2)).reshape(4,4)
> print('Example offdiagonal fermionic',f[1,2],'spin',s[1,2])
> PY`
>
> </details>
</details>

Yes. The distinction is between **a matrix acting on a tensor factor** and **a fermionic operator localized in a region**. Once you choose an ordering of the modes, you can represent both using ordinary matrices—but the correspondence includes parity strings.

There is also a qualification to my earlier statement: **absorbing the strings into a boundary factor works directly for operators of definite fermion parity.** The notebook optimizes unrestricted matrices, so that interpretation needs care.

Let me work through the convention used here.

For two ordered regions \(L,R\), write their occupation-space representation as
\[
\mathcal H_L\otimes\mathcal H_R.
\]
Matrices acting on different factors commute:
\[
[A\otimes I,I\otimes B]=0.
\]
Physical odd fermionic operators in disjoint regions instead anticommute. Therefore, they cannot both be represented simply as \(A\otimes I\) and \(I\otimes B\).

Using a Jordan–Wigner convention with strings extending to the right, a homogeneous operator \(A\) localized in \(L\) is represented by
\[
\widehat A_L=A\otimes P_R^{p(A)},
\qquad
P_LAP_L=(-1)^{p(A)}A,
\]
where \(p(A)=0\) for an even operator and \(p(A)=1\) for an odd operator. An operator localized in the rightmost region has representation
\[
\widehat B_R=I\otimes B.
\]
For odd \(A,B\), the relation \(P_RB=-BP_R\) supplies the required anticommutation. This distinction between tensor-factor embeddings and fermionic embeddings is discussed systematically in [*Fermionic systems for quantum information people*](https://arxiv.org/abs/2006.03087).

Consequently,
\[
\widehat A_L\widehat B_R
=A\otimes P_R^{p(A)}B.
\]
**The ordinary Kronecker product \(A\otimes B\) is generally not the matrix of the physical product of the two fermionic operators with those same local matrices.**

For example, with one mode in each region, let \(X\) be the local Majorana matrix and \(Z\) the local parity matrix. The physical Majoranas are represented by
\[
\gamma_L=X\otimes Z,\qquad
\gamma_R=I\otimes X,
\]
so
\[
\gamma_L\gamma_R=X\otimes ZX,
\]
rather than \(X\otimes X\).

Now consider the notebook’s three regions, ordered spatially as \(L,I,R\). Its second partial trace is
```python
symmetry_and_defect_basis.partial_trace(
    symmetry_op.matmat(rho),
    sub_sys_A=rdm_symmetry_sites,
    return_rdm='B'
)
```
Here subsystem \(A\) is the middle interval \(I\), and subsystem \(B\) contains the two exterior regions \(L,R\).

To perform this trace, QuSpin groups the modes into the order
\[
(L,I,R)\longrightarrow(I,L,R).
\]
This is a **fermionic permutation**, not just a transpose of array axes. Moving an occupied mode past another occupied mode contributes a minus sign. For occupation configurations \(l,i,r\), the rearrangement contributes
\[
(-1)^{N_L(l)N_I(i)}.
\]

I checked the installed implementation: its partial-trace routine explicitly computes these fermionic permutation signs.

Writing \(\rho\) in the original \(L,I,R\) occupation ordering, the resulting reduced matrix has elements
\[
(\rho^f_{LR})_{lr,l'r'}
=
\sum_i
(-1)^{[N_L(l)+N_L(l')]N_I(i)}
\rho_{lir,l'ir'}.
\]
An ordinary spin partial trace would omit the sign:
\[
(\rho^{\mathrm{spin}}_{LR})_{lr,l'r'}
=
\sum_i\rho_{lir,l'ir'}.
\]

These matrices can differ even for a state with definite total parity. For a three-mode example, take
\[
|\psi\rangle
=\frac{|110\rangle+|011\rangle}{\sqrt2},
\]
with modes ordered \(L,I,R\). Every component has two fermions.

Tracing out \(I\) as a spin gives the pure state
\[
\frac{|10\rangle+|01\rangle}{\sqrt2}
\]
on \(L,R\). With the fermionic reordering above, the reduced state is instead
\[
\frac{|10\rangle-|01\rangle}{\sqrt2},
\]
up to an overall phase. I verified this example directly with QuSpin. Occupation probabilities agree, but the off-diagonal coherence changes sign.

For the notebook, there is an additional factor \(P_I=(-1)^{N_I}\), giving
\[
(E_{LR})_{lr,l'r'}
=
\sum_i
(-1)^{[1+N_L(l)+N_L(l')]N_I(i)}
\rho_{lir,l'ir'}.
\]

This makes the consequence for the optimization particularly clear. Suppose \(D_L\) has definite parity \(p\). A nonzero matrix element of \(D_L\) connects configurations whose parities differ by \(p\). Therefore,
\[
\boxed{
\operatorname{Tr}\!\left[(D_L\otimes D_R)E_{LR}\right]
=
\operatorname{Tr}\!\left[
\bigl(D_L\otimes P_I^{\,1+p}\otimes D_R\bigr)\rho
\right].
}
\]
The right-hand side uses ordinary matrices in the original \(L,I,R\) ordering.

Thus:

- For **even** \(D_L\), the ordinary occupation-space expression contains \(P_I\).
- For **odd** \(D_L\), the reordering contributes another \(P_I\), so the two middle strings cancel.

That cancellation is not losing the physical parity string. It is how its combination with odd fermionic endpoint operators appears in the chosen matrix representation.

To see precisely what “absorbed into the boundary matrix” means, define physical operators localized in \(L\) and \(R\). In the right-string convention,
\[
\widehat D_L
=D_L\otimes P_I^p\otimes P_R^p.
\]
Choose the right operator’s local matrix to be
\[
\widetilde D_R=P_R^pD_R.
\]
Then
\[
\widehat D_L\,P_I\,\widehat{\widetilde D}_R
=
D_L\otimes P_I^{1+p}\otimes D_R,
\]
which is exactly the operator appearing in the boxed identity.

Multiplying \(D_R\) by \(P_R\) preserves its unitarity and its parity:
\[
P_RD_RP_R=(-1)^qD_R
\quad\Longrightarrow\quad
P_R(P_RD_R)P_R=(-1)^q(P_RD_R).
\]
So the boundary matrix changes, but the even/odd label measured by the notebook does not. With a left-string convention, the bookkeeping puts the compensating factor on the other side instead.

The limitation is that an arbitrary optimized matrix need not have a definite parity. It decomposes as
\[
D=D^{(+)}+D^{(-)},\qquad
D^{(\pm)}=\frac12(D\pm PDP).
\]
Its even component requires no compensating \(P_R\), while its odd component does. **In general, there is no single replacement \(D_R\mapsto\widetilde D_R\) that converts this mixed-parity Kronecker product into one product of physical local fermionic operators.**

Accordingly, the notebook’s matrix optimization is well-defined, but its interpretation as finding a pair of physical boundary operators is clearest when the optimized factors have definite parity—or approximately definite parity on the state’s support. This is a substantive reason to compare optimizations constrained to the even–even and odd–odd sectors, beyond simply inspecting the unconstrained parity indices.