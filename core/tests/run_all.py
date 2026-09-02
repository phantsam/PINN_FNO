"""Run the full verification suite.  Exit code 0 == the problem is verified."""
import subprocess, sys
mods = ["core.tests.test_spec", "core.tests.test_mms", "core.tests.test_physics",
        "core.tests.test_losses", "core.tests.test_metrics", "core.tests.test_models",
        "core.tests.test_train", "core.tests.test_diagnostics",
        "core.tests.test_coords", "core.tests.test_kan_variants"]
fail = 0
for m in mods:
    print("\n" + "=" * 78); print(f"  {m}"); print("=" * 78)
    r = subprocess.run([sys.executable, "-m", m], capture_output=True, text=True)
    print(r.stdout.rstrip())
    if r.returncode != 0 or "FAIL" in r.stdout or "FAILURES" in r.stdout:
        fail += 1
        if r.stderr: print(r.stderr[-800:])
print("\n" + "=" * 78)
print("VERIFICATION SUITE: " + ("ALL GREEN" if fail == 0 else f"{fail} MODULE(S) FAILED"))
sys.exit(1 if fail else 0)
