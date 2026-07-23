param(
    [ValidateSet("scalability", "compression", "noniid", "real", "all")]
    [string]$Suite = "scalability",
    [int]$Workers = [Math]::Min(8, [Environment]::ProcessorCount),
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$arguments = @(
    "-u", "run_experiments.py",
    "--suite", $Suite,
    "--workers", $Workers
)
if ($Quick) {
    $arguments += "--quick"
}
python @arguments
