import matplotlib.pyplot as plt
import numpy as np

plt.style.use('fivethirtyeight')

a = np.genfromtxt('../omp_timing_all_5h300.txt')
nmm = np.genfromtxt('../omp_timing_nmm_5h300.txt')
bmm = np.genfromtxt('../omp_timing_blockmm_5h300.txt')
nadd = np.genfromtxt('../omp_timing_nadd_5h300.txt')
batch = np.genfromtxt('../omp_timing_batching_5h300.txt')

#x = np.arange(len(a)) +1
x = [1,2,4,8,16,32,64,128,256,512]
serial_batch = [205,180,157,146,99]
fig, ax = plt.subplots()
ax.loglog(x,batch, label="16 CPU")
ax.loglog(x[-5:], serial_batch, label="1 CPU")
ax.set_xscale('log', basex=2)
#ax.semilogx(x,batch)
#ax.plot(x[4:],batch[4:])
#ax.plot(x,a, label="Fully Optimized")
#ax.plot(x,a, label="Naive Matrix Multiply")


#ax.plot(x,nmm, label="No Matrix Multiply")
#ax.plot(x, nadd, label="No Vector Add or MM")
#ax.plot(x, pmap, label="No Map Function of VA or MM")
##ax.plot(x, bmm/1000, label="Cache Optimized Matrix Multiply")
#ax.axvline(x=16, color='k', linestyle='-', alpha=0.2)

plt.legend()
#ax.set_title("OpenMP Ablation Study")
ax.set_title("Time vs Batch Size")
#ax.set_title("Time vs Number of Threads")
#ax.set_title("Comparison of Matrix Multiply")
ax.set_ylabel("Time (s)")
#ax.set_xlabel("Number of Threads")
ax.set_xlabel("Batch Size")
ax.set_xticks(x)

plt.show()
