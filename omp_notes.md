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


# Output from gprof for serial version
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  ms/call  ms/call  name    
 12.99      0.63     0.63 232252000     0.00     0.00  std::vector<std::vector<float, std::allocator<float> >, std::allocator<std::vector<float, std::allocator<float> > > >::operator[](unsigned long)
  6.80      0.96     0.33 19220148     0.00     0.00  std::move_iterator<float*> std::__make_move_if_noexcept_iterator<float, std::move_iterator<float*> >(float*)
  4.33      1.17     0.21 130682022     0.00     0.00  std::vector<float, std::allocator<float> >::size() const
  3.92      1.36     0.19 156813500     0.00     0.00  std::vector<float, std::allocator<float> >::operator[](unsigned long)
    ===
  2.68      1.49     0.13   170500     0.00     0.00  math_funcs::matrix_mult(std::vector<std::vector<float, std::allocator<float> >, std::allocator<std::vector<float, std::allocator<float> > > >, std::vector<std::vector<float, std::allocator<float> >, std::allocator<std::vector<float, std::allocator<float> > > >, std::vector<std::vector<float, std::allocator<float> >, std::allocator<std::vector<float, std::allocator<float> > > >&)
    ===

# Setting 3 -> 30
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  ms/call  ms/call  name    
 35.72     16.80    16.80 8638243000     0.00     0.00  std::vector<std::vector<float, std::allocator<float> >, std::allocator<std::vector<float, std::allocator<float> > > >::operator[](unsigned long)
    ===
 16.12     24.38     7.58   170500     0.04     0.19  math_funcs::matrix_mult(std::vector<std::vector<float, std::allocator<float> >, std::allocator<std::vector<float, std::allocator<float> > > >, std::vector<std::vector<float, std::allocator<float> >, std::allocator<std::vector<float, std::allocator<float> > > >, std::vector<std::vector<float, std::allocator<float> >, std::allocator<std::vector<float, std::allocator<float> > > >&)
    ===
 15.75     31.79     7.41 6396648500     0.00     0.00  std::vector<float, std::allocator<float> >::operator[](unsigned long)
  7.46     35.30     3.51 2483677125     0.00     0.00  std::vector<float, std::allocator<float> >::size() const
  3.51     36.95     1.65   186000     0.01     0.01  std::vector<std::vector<float, std::allocator<float> >, std::allocator<std::vector<float, std::allocator<float> > > >::operator=(std::vector<std::vector<float, std::allocator<float> >, std::allocator<std::vector<float, std::allocator<float> > > >&&)


# After fixing matrix multiply order (ikj instead of ijk)
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  ms/call  ms/call  name    
 33.34      0.05     0.05                             void std::vector<float, std::allocator<float> >::emplace_back<float&>(float&)
 13.33      0.07     0.02 20820465     0.00     0.00  std::vector<float, std::allocator<float> >::operator=(std::vector<float, std::allocator<float> > const&)
  6.67      0.08     0.01 13912408     0.00     0.00  std::_Vector_base<float, std::allocator<float> >::_M_create_storage(unsigned long)
  6.67      0.09     0.01  1922074     0.00     0.00  std::vector<float, std::allocator<float> >* std::__uninitialized_copy_a<std::move_iterator<std::vector<float, std::allocator<float> >*>, std::vector<float, std::allocator<float> >*, std::vector<float, std::allocator<float> > >(std::move_iterator<std::vector<float, std::allocator<float> >*>, std::move_iterator<std::vector<float, std::allocator<float> >*>, std::vector<float, std::allocator<float> >*, std::allocator<std::vector<float, std::allocator<float> > >&)
  6.67      0.10     0.01   964137     0.00     0.00  void std::_Destroy<__gnu_cxx::__normal_iterator<float*, std::vector<float, std::allocator<float> > >, float>(__gnu_cxx::__normal_iterator<float*, std::vector<float, std::allocator<float> > >, __gnu_cxx::__normal_iterator<float*, std::vector<float, std::allocator<float> > >, std::allocator<float>&)
  6.67      0.11     0.01   419974     0.00     0.00  int* std::__copy_move<false, true, std::random_access_iterator_tag>::__copy_m<int>(int const*, int const*, int*)
    ===
  6.67      0.12     0.01                             math_funcs::matrix_transpose(std::vector<std::vector<float, std::allocator<float> >, std::allocator<std::vector<float, std::allocator<float> > > >)
    ===
  6.67      0.13     0.01                             std::_Vector_base<float, std::allocator<float> >::_M_get_Tp_allocator() const
  3.33      0.14     0.01   543676     0.00     0.00  std::_Vector_base<int, std::allocator<int> >::_M_allocate(unsigned long)
  3.33      0.14     0.01    34119     0.00     0.00  std::_Vector_base<int, std::allocator<int> >::~_Vector_base()
  3.33      0.15     0.01     3100     0.00     0.00  int* std::__uninitialized_default_n<int*, unsigned long>(int*, unsigned long)
  3.33      0.15     0.01                             std::_Vector_base<int, std::allocator<int> >::_Vector_base()
  0.00      0.15     0.00 36149979     0.00     0.00  __gnu_cxx::__alloc_traits<std::allocator<float> >::_S_select_on_copy(std::allocator<float> const&)
  0.00      0.15     0.00 18072255     0.00     0.00  float* std::__copy_move<false, true, std::random_access_iterator_tag>::__copy_m<float>(float const*, float const*, float*)
  0.00      0.15     0.00 15824034     0.00     0.00  std::_Vector_base<float, std::allocator<float> >::_Vector_base(unsigned long, std::allocator<float> const&)
  0.00      0.15     0.00 15369623     0.00     0.00  std::allocator_traits<std::allocator<int> >::deallocate(std::allocator<int>&, int*, unsigned long)
  0.00      0.15     0.00 15041320     0.00     0.00  __gnu_cxx::__alloc_traits<std::allocator<std::vector<float, std::allocator<float> > > >::_S_select_on_copy(std::allocator<std::vector<float, std::allocator<float> > > const&)
  0.00      0.15     0.00 14225007     0.00     0.00  float* std::__copy_move_a<false, float const*, float*>(float const*, float const*, float*)
  0.00      0.15     0.00 13253522     0.00     0.00  std::_Vector_base<std::vector<float, std::allocator<float> >, std::allocator<std::vector<float, std::allocator<float> > > >::_Vector_impl::_Vector_impl()
  0.00      0.15     0.00 10841795     0.00     0.00  void std::_Construct<std::vector<float, std::allocator<float> >, std::vector<float, std::allocator<float> > const&>(std::vector<float, std::allocator<float> >*, std::vector<float, std::allocator<float> > const&)
  0.00      0.15     0.00 10772222     0.00     0.00  __gnu_cxx::__normal_iterator<float*, std::vector<float, std::allocator<float> > > std::__copy_move_a2<false, __gnu_cxx::__normal_iterator<float const*, std::vector<float, std::allocator<float> > >, __gnu_cxx::__normal_iterator<float*, std::vector<float, std::allocator<float> > > >(__gnu_cxx::__normal_iterator<float const*, std::vector<float, std::allocator<float> > >, __gnu_cxx::__normal_iterator<float const*, std::vector<float, std::allocator<float> > >, __gnu_cxx::__normal_iterator<float*, std::vector<float, std::allocator<float> > >)
  0.00      0.15     0.00 10195544     0.00     0.00  float* std::__uninitialized_copy<true>::__uninit_copy<float*, float*>(float*, float*, float*)
  0.00      0.15     0.00  9574938     0.00     0.00  std::vector<float, std::allocator<float> >::vector(std::vector<float, std::allocator<float> > const&)
  0.00      0.15     0.00  9096517     0.00     0.00  __gnu_cxx::new_allocator<std::vector<float, std::allocator<float> > >::deallocate(std::vector<float, std::allocator<float> >*, unsigned long)
  0.00      0.15     0.00  9051022     0.00     0.00  std::_Vector_base<float, std::allocator<float> >::_Vector_impl::_Vector_impl(std::allocator<float> const&)
