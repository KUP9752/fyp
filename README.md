# Final Year Project
Imperial College MEng Individual Project



## RLBench
### Windows
**DOES NOT WORK ON WINDOWS**
To run on windows I installed `CoppeliaSim` (v4 is needed I think)

export the CoppeliaSim binary location as an env variable
(Following done in Powershell)
```bash
PS > $env:COPPELIASIM_ROOT="C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu"
```


```bash
pip install git+https://github.com/stepjam/RLBench.git
```

issue with CoppeliaSim.lib (should be .dll on win) and rewriting the same symlinked file. Just going to migrate to wsl
### Ubuntu (22.04)