# Misc

  - [ ] clean up code:
    - [ ] merge parent-child classes in same file.
    - [ ] remove unsed subroutines.
    - [ ] Model_base and children.
    - [ ] Cost_base and children.
    - [ ] Controller_base.

  Convenction:
    private attribute: self._oneTwoThree.
    private methods: _one_two_three.
    public methods: one_two_three.
    class name: OneTwoThree.
    functions: one_two_three.

  - [X] Move files in dedicated folder.
  - [ ] remap imports.

  - [ ] refactor code:
    - [ ] consistent function nameing.
    - [ ] consistent private/public method nameing.
    - [ ] consistent attribut nameing.

    - Classes:
        - [X] MPPI-node.
        - [X] ModelBase.
        - [X] PointMassModel.
        - [X] AUVModel.
        - [X] NNModel.
        - [X] NNAUVModel.

        - [ ] CostBase.
        - [ ] StaticCost.
        - [ ] ElispseCost.
        - [ ] WaypointCost.

        - [ ] LearnerBase.

        - [ ] test.py
        - [ ] utile.py

        - [ ] ControllerBase.

        - [ ] Mujoco folder (deprecate)
            - [ ] genConfig (deprecate)
            - [ ] save_rng_sim (deprecate)
            - [ ] simulation (deprecate)

        - [ ] cost.py
        - [ ] main.py
        - [ ] model.py

  - [ ] Documentation:
    - [ ] costs classes.
    - [ ] model classes.
    - [ ] controller class.