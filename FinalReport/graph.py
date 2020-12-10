import matplotlib.pyplot as plt
import numpy as np

plt.style.use('fivethirtyeight')

a = np.genfromtxt('../omp_timing_all_5h300.txt')
nmm = np.genfromtxt('../omp_timing_nmm_5h300.txt')
nadd = np.genfromtxt('../omp_timing_nadd_5h300.txt')

x = np.arange(len(a))
fig, ax = plt.subplots()
ax.plot(x,a, label="Optimized")
ax.plot(x,nmm, label="No Matrix Multiply")
ax.plot(x, nadd, label="No Vector Add or MM")

plt.legend()
ax.set_title("Time vs Number of Threads")
ax.set_ylabel("Time (s)")
ax.set_xlabel("Number of Threads")

plt.show()
