# Distance-based-GP

⚠️ **Work in Progress** ⚠️

Thank you for your interest in this project! The code is currently being organized and prepared for release. Please stay tuned for upcoming updates and feel free to check back soon.

If you have any questions or suggestions, you are welcome to open an issue.

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
Run:
```bash
python main.py
```
