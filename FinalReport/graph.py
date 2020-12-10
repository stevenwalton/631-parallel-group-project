import matplotlib.pyplot as plt
import numpy as np

plt.style.use('fivethirtyeight')

a = np.genfromtxt('../omp_timing_all_5h300.txt')
nmm = np.genfromtxt('../omp_timing_nmm_5h300.txt')
bmm = np.genfromtxt('../omp_timing_blockmm_5h300.txt')
nadd = np.genfromtxt('../omp_timing_nadd_5h300.txt')

x = np.arange(len(a)) +1
fig, ax = plt.subplots()
#ax.plot(x,a, label="Fully Optimized")
ax.plot(x,a, label="Naive Matrix Multiply")

#ax.plot(x,nmm, label="No Matrix Multiply")
#ax.plot(x, nadd, label="No Vector Add or MM")
ax.plot(x, bmm/1000, label="Cache Optimized Matrix Multiply")
ax.axvline(x=16, color='k', linestyle='-', alpha=0.2)

plt.legend()
#ax.set_title("OpenMP Ablation Study")
#ax.set_title("Time vs Number of Threads")
ax.set_title("Comparison of Matrix Multiply")
ax.set_ylabel("Time (s)")
ax.set_xlabel("Number of Threads")

plt.show()
