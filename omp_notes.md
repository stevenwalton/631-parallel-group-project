| method | time |
|:------:|:----:|
| Parallelize outer updateWeights loop set nodes to 1k 5 epochs (32 threads)| 39.046s | 
| Parallelize outer updateWeights loop set nodes to 1k 5 epochs (4 threads)| 41.985s | 
| Parallelize inner updateWeights loop set nodes to 1k 5 epochs (32 threads)| 60.72s | 
| set nodes to 1k 5 epochs| 41.985s |

# What worked

# What didn't work
- Parallelizing weight updates doesn't do much. Node sizes need to be above 1k
for there to be any noticeable difference.
- Parallelizing the outer loop of updateWeights is significantly faster than the
parallelizing the inner loop
