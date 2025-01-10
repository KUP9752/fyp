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

### Ubuntu VM (24.04)

It did not want to run on a VM either, I had some graphics issues and copeelia sim refused to work on a VM

### Ubuntu Dual Boot (24.04)

Worked completely seamlessly other than a video compression library missing warning every once in a while

- Tells me there might be some `lib...-dev` libraries missing, when I try to resolve, I see that they are indeed installed. This doesn't seem to be an error though, will check up on if it causes issues.
