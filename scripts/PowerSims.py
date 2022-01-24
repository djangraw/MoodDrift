import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.weightstats import ttost_ind
from matplotlib import pyplot as plt

rng = np.random.default_rng(1)

# calc power assuming null effect
nsims = 10000
es = 0
tost_alpha = 0.01
sim_res = []
for n in [145, 150, 155, 190, 195, 200, 250, 300, 400, 500]:
    for dt in [0.2, 0.3, 0.35, 0.4, 0.5]:
        ldt = -1 * dt
        udt = dt
        for sn in range(nsims):
            a_dat = rng.standard_normal(n)
            b_dat = rng.standard_normal(n) + es
            sim_row = {}
            sim_row['sim_n'] = sn
            sim_row['n'] = n
            sim_row['dt'] = dt
            sim_row['md'] = a_dat.mean()-b_dat.mean()
            es_denom = np.sqrt((((n-1) * a_dat.var()) + ((n-1) * b_dat.var())) / ((2 * n) - 2))
            sim_row['d'] = sim_row['md']/es_denom
            sim_row['bmean'] = b_dat.mean()
            sim_row['amean'] = a_dat.mean()
            sim_row['tstat'], sim_row['p'] = stats.ttest_ind(a_dat, b_dat)
            # calculate tost
            sim_row['tost_p'], _, _ = ttost_ind(a_dat, b_dat, ldt * es_denom, udt * es_denom)
            sim_res.append(sim_row)
sim_res = pd.DataFrame(sim_res)


sim_res['tost_sig'] = sim_res.tost_p < tost_alpha
sim_res['abs_d'] = np.abs(sim_res.d)
print("Power assuming that the true effect is null:")
print(sim_res.groupby(['n', 'dt']).tost_sig.mean())


# calc power in the case that true effect is not null

sim_res = []
pop_n = 10000
effect_sizes = rng.uniform(-0.5, 0.5, size=100000)
#for n in [150, 300, 500]:
ns = [150, 300, 500]
for dt in [0.3, 0.5]:
    ldt = -1 * dt
    udt = dt
    for sn, es in enumerate(effect_sizes):
        pop = rng.standard_normal((2,pop_n))
        pop_a = pop[0,:]
        pop_b = pop[1,:] + es
        pop_md = pop_b.mean() - pop_a.mean()
        pop_es_denom = np.sqrt((((n-1) * pop_a.var()) + ((n-1) * pop_b.var())) / ((2 * n) - 2))

        for n in ns:
            a_dat = rng.choice(pop_a, size=n, replace=False)
            b_dat = rng.choice(pop_b, size=n, replace=False)
            sim_row = {}
            sim_row['sim_n'] = sn
            sim_row['n'] = n
            sim_row['dt'] = dt
            sim_row['pop_md'] = pop_md
            sim_row['samp_md'] = a_dat.mean()-b_dat.mean()
            es_denom = np.sqrt((((n-1) * a_dat.var()) + ((n-1) * b_dat.var())) / ((2 * n) - 2))
            sim_row['d'] = sim_row['samp_md']/es_denom
            sim_row['pop_d'] = sim_row['pop_md']/pop_es_denom
            sim_row['abs_pop_d'] = np.abs(sim_row['pop_md']/pop_es_denom)

            sim_row['bmean'] = b_dat.mean()
            sim_row['amean'] = a_dat.mean()
            sim_row['tstat'], sim_row['p'] = stats.ttest_ind(a_dat, b_dat)
            # calculate tost
            sim_row['tost_p'], _, _ = ttost_ind(a_dat, b_dat, ldt * es_denom, udt * es_denom)
            sim_res.append(sim_row)
sim_res = pd.DataFrame(sim_res)
sim_res['tost_sig'] = sim_res.tost_p < tost_alpha
sim_res['abs_d'] = np.abs(sim_res.d)
sim_res['abs_pop_d_bins'] = pd.cut(sim_res.abs_pop_d, 100)


sim_res['dummy'] = 1
cp_stab = []
fig, ax = plt.subplots(1)
for n in sim_res.n.unique():
    for dt in sim_res.dt.unique():
        for stbn in range(100):
            sim_res_cp = []
            for ix, df in sim_res.query("abs_pop_d < 0.5 & dt == @dt & n == @n").groupby(['abs_pop_d_bins']):
                try:
                    sim_res_cp.append(df.sample(n=1000, replace=False))
                except ValueError:
                    pass
            sim_res_cp = pd.concat(sim_res_cp)
            sim_res_cp = sim_res_cp.sort_values('abs_pop_d').reset_index(drop=True)
            sim_res_cp['cumulative_power'] = sim_res_cp.tost_sig.cumsum()/sim_res_cp.dummy.cumsum()
            # trim early value as these tend to be noisy
            sim_res_cp = sim_res_cp.query('abs_pop_d > 0.05')

            row={}
            row['dt'] = dt
            row['n'] = n
            try:
                max_cp_under_08 = sim_res_cp.query('cumulative_power <= 0.8').cumulative_power.max()
                row['abs_pop_d_08'] = sim_res_cp.loc[sim_res_cp.cumulative_power == max_cp_under_08, 'abs_pop_d'].values[0]
                row['cp_08'] = sim_res_cp.loc[sim_res_cp.cumulative_power == max_cp_under_08, 'cumulative_power'].values[0]
            except IndexError:
                pass
            max_cp_under_095 = sim_res_cp.query('cumulative_power <= 0.95').cumulative_power.max()
            row['abs_pop_d_095'] = sim_res_cp.loc[sim_res_cp.cumulative_power == max_cp_under_095, 'abs_pop_d'].values[0]
            row['cp_095'] = sim_res_cp.loc[sim_res_cp.cumulative_power == max_cp_under_095, 'cumulative_power'].values[0]
            cp_stab.append(row)
            if stbn == 0:
                ax.plot(sim_res_cp.query('abs_pop_d > 0.05').abs_pop_d, sim_res_cp.query('abs_pop_d > 0.05').cumulative_power, label=f'n={n}, dt={dt}')
cp_stab = pd.DataFrame(cp_stab)
ax.legend()
ax.set_xlabel('Population effect size')
ax.set_ylabel('Power')
fig.set_facecolor('white')
fig.savefig('../Figures/tost_cumulative_power.png')

print(cp_stab.query('dt == 0.5 & n == 150').describe())
print(cp_stab.query('dt == 0.3 & n == 150').describe())
print(cp_stab.query('dt == 0.5 & n == 300').describe())
print(cp_stab.query('dt == 0.3 & n == 300').describe())
print(cp_stab.query('dt == 0.5 & n == 500').describe())
print(cp_stab.query('dt == 0.3 & n == 500').describe())

print("calculate tost performance if effect size is actually 0.5")
es = 0.5
# calc null_perf
sim_res = []
for n in [145, 150, 155, 190, 195, 200]:
    for sn in range(nsims):
        a_dat = rng.standard_normal(n)
        b_dat = rng.standard_normal(n) + es
        sim_row = {}
        sim_row['sim_n'] = sn
        sim_row['n'] = n
        sim_row['md'] = a_dat.mean()-b_dat.mean()
        es_denom = np.sqrt((((n-1) * a_dat.var()) + ((n-1) * b_dat.var())) / ((2 * n) - 2))
        sim_row['d'] = sim_row['md']/es_denom
        sim_row['bmean'] = b_dat.mean()
        sim_row['amean'] = a_dat.mean()
        sim_row['tstat'], sim_row['p'] = stats.ttest_ind(a_dat, b_dat)
        # calculate tost
        sim_row['tost_p'], _, _ = ttost_ind(a_dat, b_dat, ldt * es_denom, udt * es_denom)
        sim_res.append(sim_row)
sim_res = pd.DataFrame(sim_res)
sim_res['tost_sig'] = sim_res.tost_p < tost_alpha
print(sim_res.groupby('n').tost_sig.mean())