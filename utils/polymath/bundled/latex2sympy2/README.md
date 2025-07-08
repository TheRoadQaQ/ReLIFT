# Bundled latex2sympy2

This package is adapted from <https://github.com/OrangeX4/latex2sympy> with several updates:

- Update ANTLR4 version to 4.11.0.
- Refactor the project folder structure.

## Generate Command

```bash
PROJECT_ROOT=/path/to/uniform-eval
OUTPUT_DIR=${PROJECT_ROOT}/src/uniform_eval/bundled/latex2sympy2/gen
antlr4 -o ${OUTPUT_DIR} -package gen -listener -encoding UTF-8 -Dlanguage=Python3 -no-visitor -lib ${PROJECT_ROOT}/src/uniform_eval/bundled/latex2sympy2/assets ${PROJECT_ROOT}/src/uniform_eval/bundled/latex2sympy2/PS.g4
rm ${OUTPUT_DIR}/*.interp
rm ${OUTPUT_DIR}/*.tokens
```
