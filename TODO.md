# MPPI-ROS node

- [X] include the controller.
- [X] load the params.
- [X] parse the config files.
- [X] Convert velocity state message from inertial to body frame.
- [ ] tune the simulation real time factor to match MPPI's execution.
- [ ] tune MPPI hyperparameters.


## Bug:

The same input produces different response when applied to the simulator vs to the predictive model. The same input produces a + x and y for the predictive model while this produces a - x and y for the simulator.

    - Check if the frame in which the input tensor is expressed is the same frame as the one expected by the thruster manager. Seems to be the same but need to look at thruster_manager.py/publish_thrust_forces.

    Thruster manager config atm is:
    '''
        tf_prefix: "/rexrov2/"
        base_link: "base_link" !! NOT BASE_LINK_NED
        thruster_frame_base: "thruster_"
        thruster_topic_prefix: "thrusters/"
        thruster_topic_suffix: "/input"
        timeout: -1.0
        max_thrust: 1540.0
        n_thrusters: 6
    '''

    The frame used when publishing to the thruster manager is also base_link

    Torque tensor seems to be expressed in body frame in controller and simulation.

    trying to compute the acceleration in SNAME formulation.


## Experiments:

  ### Fossen AUV model.

  #### Static cost:
    - [ ] lambda variation, with/without normalization.
    - [ ] change noise, from 100N-3000N.
    - [ ] augment time horizon.
    - [ ] change number of samples 1000-5000.
    - [ ] Vary cost shape.
    - [ ] change other hyperparameters.
    - [ ] add current.

  #### Eliptic cost:
    - [ ] lambda variation, with/without normalization.
    - [ ] change noise, from 100N-3000N.
    - [ ] augment time horizon.
    - [ ] change number of samples 1000-5000.
    - [ ] Vary cost shape.
    - [ ] change other hyperparameters.
    - [ ] add current.

  #### Path cost:
    - [ ] lambda variation, with/without normalization.
    - [ ] change noise, from 100N-3000N.
    - [ ] augment time horizon.
    - [ ] change number of samples 1000-5000.
    - [ ] Vary cost shape.
    - [ ] change other hyperparameters.
    - [ ] add current.

  #### Model learning:
    - [ ] learn simple parameters such as Mass, volume, density first.
    - [ ] expand to a larger set of parameters.
    - [ ] simple gradient descent technique.
    - [ ] look at other system identification techniques.

  ### NN AUV model.

  #### Model learning:
    - [ ] see if the model can learn anything.


  ### GP AUV model.