# Valence Quarks — Mathematics Engine Specification

**Proton-I stage:** 2  
**Status:** VQ-01 safe exact arithmetic + immutable AST implemented and CI-verified (partial); VQ-02 and later domains are planned  
**Purpose:** Make Valence the mathematical problem-solving component of Proton-I.

## 1. Product contract

Valence owns the complete mathematics capability of Proton-I: from elementary arithmetic through advanced university and research-level mathematics. It must do more than retrieve a web snippet: it parses a mathematical request, selects a suitable method, performs a verifiable calculation or derivation, and communicates the result.

Mathematics includes both knowledge and computation. Therefore, basic calculations such as `2 + 2` belong to Valence too. The current scope assigns all mathematical work to Valence so that the stages do not duplicate one another. Sea Quarks remains reserved for a future, separately defined capability.

Valence is a deterministic mathematics engine, not a claim that one small neural network will have learned every mathematical discipline. Use explicit algorithms, symbolic representations, numerical methods, and validation. A language model may later help interpret wording or explain results, but it must not be trusted as the numerical authority.

## 2. Mathematics coverage

The target is broad coverage, delivered incrementally. The list below defines the intended domain; it is not a claim that every item is already implemented.

### A. Arithmetic and number systems
- Natural numbers, whole numbers, integers, rational numbers, real numbers, and complex numbers.
- Addition, subtraction, multiplication, division, order of operations, parentheses, fractions, decimals, ratios, proportions, percentages, rates, and unit conversions.
- Divisibility, primes, factors, multiples, GCD, LCM, modular arithmetic, remainders, and integer properties.
- Powers, roots, scientific notation, absolute value, logarithms, exponentials, and exact versus approximate values.
- Fractions and rational expressions should remain exact where possible.

### B. Algebra
- Variables, constants, terms, expressions, identities, expansion, simplification, and factorisation.
- Linear, quadratic, polynomial, rational, radical, exponential, and logarithmic equations.
- Systems of equations, inequalities, compound inequalities, and solution sets.
- Polynomials, roots, multiplicity, remainder/factor theorems, and partial fractions.
- Functions, domain, range, composition, inverse functions, transformations, and piecewise definitions.
- Sequences, progressions, recurrence relations, and elementary series.

### C. Geometry and coordinate mathematics
- Euclidean geometry: points, lines, angles, triangles, polygons, circles, congruence, similarity, perimeter, area, surface area, and volume.
- Coordinate and analytic geometry: distance, midpoint, slope, line equations, circles, conics, and intersections.
- 2D and 3D geometry, solid geometry, transformations, loci, and geometric proofs.
- Vectors in 2D and 3D, dot product, cross product, projections, and geometric applications.

### D. Trigonometry
- Radians and degrees, unit circle, trigonometric functions, graphs, identities, and inverse trigonometric functions.
- Trigonometric equations, equations involving multiple angles, and triangle-solving rules.
- Hyperbolic functions and identities as an advanced extension.

### E. Precalculus and mathematical functions
- Function behaviour, composition, inverse functions, continuity intuition, asymptotes, and graph transformations.
- Exponential and logarithmic models, polar coordinates, parametric equations, and conic sections.
- Sequences, infinite series, convergence intuition, and Taylor/Maclaurin preparation.

### F. Calculus and analysis
- Limits, one-sided limits, continuity, derivatives, implicit differentiation, related rates, and optimisation.
- Definite and indefinite integration, substitution, integration by parts, partial fractions, and improper integrals.
- Sequences and series, convergence tests, Taylor and power series.
- Multivariable calculus: partial derivatives, gradients, Jacobians, Hessians, multiple integrals, and constrained extrema.
- Vector calculus: vector fields, line/surface integrals, divergence, curl, and integral theorems.
- Ordinary differential equations (ODEs), systems of ODEs, and introductory partial differential equations (PDEs).
- Real analysis and complex analysis at an appropriately staged level.

### G. Linear algebra
- Vectors, matrices, dimensions, matrix operations, transpose, determinant, inverse, rank, and trace.
- Linear systems, row reduction, vector spaces, bases, dimension, linear independence, and subspaces.
- Linear transformations, eigenvalues, eigenvectors, diagonalisation, orthogonality, projections, and least squares.
- Singular value decomposition (SVD), quadratic forms, and numerical conditioning.

### H. Probability and statistics
- Counting, permutations, combinations, sample spaces, events, conditional probability, Bayes' theorem, and independence.
- Random variables, expectation, variance, covariance, common discrete and continuous distributions.
- Sampling, descriptive statistics, estimation, confidence intervals, hypothesis tests, and p-values.
- Correlation, regression, residual analysis, and elementary statistical modelling.
- Statistical output must distinguish assumptions, estimates, uncertainty, and conclusions.

### I. Discrete mathematics and logic
- Propositional and predicate logic, truth tables, quantifiers, proof methods, and mathematical induction.
- Sets, relations, equivalence relations, mappings, and cardinality.
- Combinatorics, recurrence relations, graph theory, trees, paths, and networks.
- Boolean algebra, algorithms' mathematical complexity, and discrete structures.

### J. Number theory
- Divisibility, Euclidean algorithm, congruences, modular inverses, prime factorisation, and Diophantine equations.
- Elementary results on primes, residues, and integer sequences.
- More advanced algebraic and analytic number theory is a long-term extension, not an early-release claim.

### K. Numerical mathematics and optimisation
- Root finding, bisection, Newton methods, numerical differentiation, and numerical integration.
- Linear-system solvers, interpolation, approximation, error bounds, and conditioning.
- Floating-point behaviour, precision control, convergence, stability, and error reporting.
- Unconstrained and constrained optimisation, linear programming, and convex optimisation foundations.

### L. Advanced mathematical extensions
- Abstract algebra: groups, rings, fields, homomorphisms, and quotient structures.
- Topology, metric spaces, measure theory, functional analysis, and differential geometry.
- Tensor algebra and calculus, Fourier and Laplace transforms, and more advanced PDE methods.
- Information theory, game theory, and mathematical modelling where they are genuinely mathematical tasks.

## 3. System architecture

Keep the mathematics engine isolated from Gluons. Gluons continues to search websites and extract concise source-grounded answers; it must not become the calculator or symbolic solver.

Proposed package boundary:

```text
src/proton_i/valence/
    __init__.py
    engine.py             # public solve/compute interface
    intent.py             # identify the mathematical task
    lexer.py              # tokenize mathematical notation
    parser.py             # build a validated expression tree
    expressions.py        # immutable symbolic expression nodes
    numbers.py            # exact and approximate number types
    explain.py            # structured solution steps
    verify.py             # independent result checks
    domains/
        arithmetic.py
        number_theory.py
        algebra.py
        equations.py
        functions.py
        geometry.py
        trigonometry.py
        linear_algebra.py
        probability.py
        statistics.py
        discrete.py
        calculus.py
        differential_equations.py
        numerical.py
        optimisation.py
        advanced.py
```

This is a target layout, not a requirement to create all files as empty placeholders. Add modules only when their tests and implementation are ready.

### Processing pipeline

1. **Input validation:** reject empty, malformed, excessively long, or unsupported requests clearly.
2. **Task recognition:** identify the mathematical domain and requested operation; ask for clarification only when a missing condition changes the answer.
3. **Parsing:** convert supported text and mathematical notation into a structured expression or problem representation. VQ-01 builds immutable `NumberNode`, `UnaryNode`, and `BinaryNode` nodes.
4. **Planning:** select a documented exact, symbolic, or numerical method.
5. **Execution:** evaluate the expression tree with deterministic algorithms and explicit precision/resource bounds.
6. **Verification:** substitute solutions back into the original problem, check residuals, compare identities, or use an independent reference where possible.
7. **Explanation:** return the answer, meaningful intermediate steps, assumptions, domain restrictions, and approximation status.

## 4. Correctness and safety requirements

- Never use Python `eval`, `exec`, or arbitrary expression execution to calculate user input. Parse into an allowlisted expression tree and evaluate only supported node types.
- Bound input length, tree depth, iteration counts, matrix dimensions, and numerical work. Handle timeouts or unsupported complexity without hanging.
- Prefer exact integers, rational fractions, and symbolic constants when possible. Do not silently convert exact results into floating-point approximations.
- When approximation is necessary, state that it is approximate and expose the precision or tolerance used.
- Handle division by zero, invalid roots/logarithms, singular matrices, convergence failures, and undefined expressions explicitly.
- Track assumptions and domains, such as real versus complex solutions, angle units, variable constraints, and units of measurement.
- Never fabricate a derivation. If a method cannot be verified or a problem is unsupported, report that limitation.
- Separate a mathematically valid result from an explanation generated in natural language. The result must be checked by the mathematical engine.
- No network access is required for mathematical computation. Website retrieval remains Gluons' responsibility.

## 5. Output contract

For a supported problem, Valence should provide:
- The final answer.
- A concise, ordered derivation when the task asks for working or when steps are needed to make the result understandable.
- Exact form and decimal approximation when both are useful.
- Necessary domain restrictions, assumptions, units, and angle convention.
- A clear note when a result is numerical, approximate, conditional, or not fully verified.

Do not force long explanations for trivial arithmetic. Do not hide steps for multi-stage algebra, calculus, proofs, or statistics questions. Final presentation format can be refined separately; the mathematical result and verification are mandatory.

## 6. Implementation roadmap

Build and verify the engine in small milestones. Each milestone must have tests before implementation, a runnable example, and a short report. The following sequence is proposed:

| Milestone | Scope | Minimum evidence |
|---|---|---|
| VQ-01 | Safe tokenizer, recursive-descent parser, immutable expression tree, exact arithmetic, parentheses, rational decimals | *Implemented (partial)* — exact hand-calculated cases, AST precedence/immutability checks, malicious-input rejection, and full CI verification; no powers, roots, or symbolic algebra |
| VQ-02 | Number systems, powers, roots, logarithms, percentages, ratios, number properties | In progress next — bounded integer powers, exact square roots where possible, GCD/LCM, percentages, ratios, and explicit domain/error tests |
| VQ-03 | Symbolic expressions, simplification, expansion, factorisation | Identity checks and canonical-form tests |
| VQ-04 | Equations, inequalities, systems, polynomials | Substitution/residual checks; no-solution and infinite-solution cases |
| VQ-05 | Functions, graphs' mathematical definitions, sequences, series, precalculus | Domain/range and known-value tests |
| VQ-06 | Geometry, coordinate geometry, 2D/3D vectors | Formula checks, unit tests, geometric invariants |
| VQ-07 | Trigonometry and hyperbolic functions | Identity, periodicity, angle-unit, and inverse-function tests |
| VQ-08 | Linear algebra | Shape validation and comparisons against trusted NumPy calculations |
| VQ-09 | Probability and statistics | Exact probability cases, seeded experiments, and reference calculations |
| VQ-10 | Single-variable limits, derivatives, and integration | Analytical examples, substitution checks, finite-difference/reference comparisons |
| VQ-11 | Series, multivariable/vector calculus, ODEs and introductory PDEs | Residual checks and convergence/error tests |
| VQ-12 | Discrete mathematics, logic, combinatorics, graph theory, and number theory | Exhaustive small-case checks and known theorems |
| VQ-13 | Numerical analysis and optimisation | Error bounds, convergence tests, stability checks, known optima |
| VQ-14 | Advanced extensions and broader natural-language problem interpretation | Domain-specific proof/reference suites and explicit capability matrix |

Milestones are sequential. Do not skip ahead or claim full mathematics coverage after an early milestone.

## 7. Technology and dependency policy

- Start with the existing Python package and current NumPy dependency. Use the Python standard library for exact integers, rational arithmetic, decimal arithmetic, parsing utilities, and core algorithms where appropriate.
- Use NumPy for numerical arrays and matrix operations where appropriate; compare results with independent hand-derived cases.
- Do not add SymPy, SciPy, mpmath, an LLM API, or another major dependency without the owner's approval. Such tools may be evaluated as optional references or backends after their licensing, size, and role are reviewed.
- Do not use a machine-learning model as the source of truth for arithmetic or symbolic evaluation.
- Keep Valence's runtime local and deterministic in its first implementation; any hosted or remote execution design is a separate decision.

## 8. Definition of done

Valence is not complete merely because it can solve a few sample equations. A release must publish a capability matrix that marks each domain as supported, partial, experimental, or unsupported. Every supported operation must have:
- Unit and edge-case tests with independent expected results.
- Validation of undefined inputs and numerical failure modes.
- Property/invariant tests where applicable.
- A runnable example and documented input syntax.
- Explicit precision, approximation, and domain behaviour.
- Full repository test-suite evidence.

A domain must remain marked partial until its stated coverage has been implemented and tested. Unsupported questions must receive an honest limitation rather than an invented answer.
