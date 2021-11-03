# Profiling mppi. 

 Write everything in the controllerBase class. The profiling will be as structure of nested dictionnary. A "sub" dictionnary indicates a sub routine called withing the parent dict. 
 For the AUV we would have:
```python
    profile = {'total': step_timing,
               'random': rand_timing, 
               'rollout': {'total': rollout_timing,
                           'cost': cost_timing,
                           'model': {'total': model_step_timing,
                                     'b2i_transform': b2i_timing,
                                     'pose_dot': pd_timing,
                                     'acc': {'total': acc_total_timing,
                                             'damping': d_timing,
                                             'coriolis': c_timing,
                                             'restoring': r_timing,
                                             'solv': s_timing,
                                             'calls': acc_calls
                                            }
                                     'calls': state_dot_calls
                                    },
                           'calls': rollout_calls
                           'horizon': nb_horizon
                          },
               'update': update_timing,
               'calls': controller_calls}
```

 - [ ] generate the profiling data: 
    - [X] step timing.
    - [X] random number timing.
    - [ ] rollout timing:
        - [X] total timing.
        - [X] cost timing.
        - [ ] body2inertial_transform.
        - [ ] pose_dot.
        - [ ] acc:
            - [ ] Damping.
            - [ ] Coriolis.
            - [ ] Restoring.
    - [X] Update.
 - [ ] Visualization of the profiling.

 - [ ] Get data for the pc.
 - [ ] Get data for the GPU-server.