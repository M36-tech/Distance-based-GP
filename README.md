<!-- # Distance-based-GP
This repository contains the ​​Python implementation​​ of Distance-based-GP​ from the paper:

​​"Developing distance-based Genetic Programming classifiers by reconstructing datasets for imbalanced binary classification"​

The current code is largely complete and functional. We may continue to make improvements and optimizations in future versions.

If you have any questions or suggestions, you are welcome to open an issue. -->


# Distance-based-GP

This repository provides the Python implementation of **Distance-based-GP** proposed in the paper:

> **"Developing distance-based Genetic Programming classifiers by reconstructing datasets for imbalanced binary classification"**

The current code is largely complete and functional. We may continue to make improvements and optimizations in future versions.

If you have any questions or suggestions, feel free to open an issue.


## Installation

```bash
pip install -r requirements.txt
```

Replace the compile function in deap.gp with:
```python
######################################
# GP Tree compilation functions      #
######################################
def compile(expr, pset):
    """Compile the expression *expr*.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                a string of Python code or any object that when
                converted into string produced a valid Python code
                expression.
    :param pset: Primitive set against which the expression is compile.
    :returns: a function if the primitive set has 1 or more arguments,
            or return the results produced by evaluating the tree.
    """
    code = str(expr)
    if len(pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        # args = ",".join(arg for arg in pset.arguments)
        for f in reversed(pset.arguments):
            if code.find(f) != -1:
                index = f[1:]
                replace_str = "list" + "[" + index + "]"
                # print(str)
                code = code.replace(f, replace_str)
        code = "lambda list: {code}".format(code=code)
    try:
        # print(code)
        return eval(code, pset.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("DEAP : Error in tree evaluation :"
                            " Python cannot evaluate a tree higher than 90. "
                            "To avoid this problem, you should use bloat control on your "
                            "operators. See the DEAP documentation for more information. "
                            "DEAP will now abort.").with_traceback(traceback)
```
## Run:
```bash
python main.py
```
If you find this repository is useful for you, please cite our paper:
```bibtex
@article{MENG2026112825,
title = {Developing distance-based genetic programming classifiers by reconstructing datasets for imbalanced binary classification},
journal = {Pattern Recognition},
volume = {173},
pages = {112825},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.112825},
url = {https://www.sciencedirect.com/science/article/pii/S0031320325014888},
author = {Wenyang Meng and Ying Li and Fan Zhang and Xiaoying Gao and Jianbin Ma},
keywords = {Genetic programming, Distance, Dataset reconstructing, Imbalanced binary classification}
}
```

