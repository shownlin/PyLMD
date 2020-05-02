# PyLMD
Method of decomposing signal into Product Functions

This project implements the paper:

[Jonathan S. Smith. The local mean decomposition and its application to EEG perception data. Journal of the Royal Society Interface, 2005, 2(5):443-454](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1618495/)

## How to install?

```
pip install PyLMD
```

requirements:
1. numpy
2. scipy

## Examples
```python
>>> import numpy as np
>>> from PyLMD import LMD
>>> x = np.linspace(0, 100, 101)
>>> y = 2 / 3 * np.sin(x * 30) + 2 / 3 * np.sin(x * 17.5) + 4 / 5 * np.cos(x * 2)
>>> lmd = LMD()
>>> PFs, resdue = lmd.lmd(y)
>>> PFs.shape
(6, 101)
```

![Example](https://raw.githubusercontent.com/shownlin/PyLMD/master/simple_example.png)

## Parameters
* **INCLUDE_ENDPOINTS** : bool, (default: True)

    Whether to treat the endpoint of the signal as a pseudo-extreme point

* **max_smooth_iteration** : int, (default: 12)

    Maximum number of iterations of moving average algorithm.

* **max_envelope_iteration** : int, (default: 200)

    Maximum number of iterations when separating local envelope signals.

* **envelope_epsilon** : float, (default: 0.01)

    Terminate processing when obtaining pure FM signal.

* **convergence_epsilon** : float, (default: 0.01)

    Terminate processing when modulation signal converges.

* **max_num_pf** : int, (default: 8)

    The maximum number of PFs generated.

## Return
* **PFs** : numpy array

    The decompose functions arrange is arranged from high frequency to low frequency.

* **residue** : numpy array
    
    residual component

## Contact
Use GitHub Issues or the mailing list to post your comments or questions.


## License
PyLMD is a free Open Source software released under the MIT license.


Lin, Q., 2020. **Python Implementation Of Local Mean Decomposition Algorithm.**
(https://github.com/shownlin/PyLMD)