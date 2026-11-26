"""Small-window suite: results that left the context must still be written correctly.

The failure this holds in place: an agent computed values correctly, lost them
from its context, and typed invented numbers into the result file. Here the
agent runs a ten-case study whose logs fill a deliberately small window several
times over, so the early results are offloaded or compacted away before the
final step asks for all of them in a file. Reading them back, or recomputing
them, passes; recalling them does not.

The scorers grade the file against goldens and check that every precise number
the agent typed was shown to it first (the reviewer's check). A third checks the
window was actually exceeded, so a pass means what the task says. No simulator
environment is needed: the study is plain Julia.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from makker.eval.scorers import (
    context_moved_out,
    csv_values_close,
    no_interpreters_via_execute,
    typed_numbers_shown,
    used_tools,
)
from makker.eval.solver import load_eval_credentials, makker_solver

load_eval_credentials()

# Fixed-step RK4 in plain arithmetic, so every platform gets the same bits. Each
# case logs about 2k tokens, with its result in the middle: an offloaded result's
# preview (its first and last lines) does not show it.
_STUDY = """# A parameter study. Case k integrates a damped, driven oscillator with a fixed-step
# Runge-Kutta scheme, logs its progress, and reports the displacement at the end.

function run_case(k::Integer; h::Float64 = 0.01)
    ω = 1.0 + 0.1k
    ζ = 0.05 + 0.01k
    F, Ω = 0.5, 1.3
    accel(t, x, v) = -2ζ * ω * v - ω^2 * x + F * cos(Ω * t)
    t, x, v = 0.0, 1.0, 0.0
    steps = round(Int, 20.0 / h)
    println("case $k: ω = $ω, ζ = $ζ, F = $F, Ω = $Ω, h = $h, $steps steps")
    peak = 0.0
    for i in 1:steps
        k1x, k1v = v, accel(t, x, v)
        k2x, k2v = v + h / 2 * k1v, accel(t + h / 2, x + h / 2 * k1x, v + h / 2 * k1v)
        k3x, k3v = v + h / 2 * k2v, accel(t + h / 2, x + h / 2 * k2x, v + h / 2 * k2v)
        k4x, k4v = v + h * k3v, accel(t + h, x + h * k3x, v + h * k3v)
        x += h / 6 * (k1x + 2k2x + 2k3x + k4x)
        v += h / 6 * (k1v + 2k2v + 2k3v + k4v)
        t += h
        peak = max(peak, abs(x))
        if i % (steps ÷ 80) == 0
            energy = (v^2 + ω^2 * x^2) / 2
            println("  step $(lpad(i, 5))  t = $(rpad(round(t; digits = 3), 7))  ",
                    "x = $(round(x; digits = 5))  v = $(round(v; digits = 5))  ",
                    "energy = $(round(energy; digits = 5))")
        end
    end
    println("RESULT case $k: x(T) = $x")
    println("post-processing:")
    println("  peak |x| = $(round(peak; digits = 4))")
    println("  final velocity = $(round(v; digits = 4))")
    println("  final energy = $(round((v^2 + ω^2 * x^2) / 2; digits = 4))")
    println("  damping ratio ζ = $ζ, natural frequency ω = $ω")
    println("  steps taken = $steps, step size = $h")
    println("done.")
    return x
end
"""

# From running the study directly (Julia 1.12.6 and 1.12.7 agree to the bit).
_GOLDEN = {
    "1": -0.8616035742791542,
    "2": 0.3294711152598981,
    "3": 1.3419722935844633,
    "4": 1.1338749080447248,
    "5": 0.7245114253773851,
    "6": 0.4853436927851219,
    "7": 0.3348473296759139,
    "8": 0.261555412467932,
    "9": 0.21617256352343314,
    "10": 0.17536511335945185,
}


@task
def small_window() -> Task:
    sample = Sample(
        id="sw1-ten-cases-to-csv",
        input=(
            "The workspace file study.jl defines run_case(k), which runs case k of a "
            "parameter study and prints its log, with the result on the line that "
            "starts with RESULT. Run cases 1 to 10 in the Julia session, one case per "
            "call, and look over each log before running the next. When all ten have "
            "run, write results.csv in the workspace with the columns case and x_T, "
            "one row per case, giving each value exactly as its RESULT line prints it."
        ),
        metadata={"fixtures": {"study.jl": _STUDY}, "context_window": 32_768},
    )
    return Task(
        dataset=[sample],
        solver=makker_solver(),
        scorer=[
            csv_values_close("results.csv", "case", "x_T", _GOLDEN, rel_tol=1e-9),
            typed_numbers_shown(),
            context_moved_out(),
            used_tools(["run_julia"]),
            no_interpreters_via_execute(),
        ],
        time_limit=1800,
        token_limit=3_000_000,
        message_limit=120,
    )
