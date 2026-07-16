# Coverage Policy

## CI Behavior
- The `coverage` job runs on every push and produces `coverage.xml`.
- The diff-cover gate runs on pull requests only.

## Ratchet Rules
- Patch threshold: new or changed lines in CPU-testable paths must achieve `>=80%` coverage. This is enforced by diff-cover on every PR.
- Increment rule: raise `fail_under` by `+1` to `+2` percent at a time, after a climb wave lands and the combined baseline is stably above the current floor for 2+ CI runs. Never in a hurry.
- Never lower rule: the global floor (`fail_under` in the CI coverage job) must never be lowered. If coverage temporarily dips due to a large refactor, fix the tests rather than lowering the floor.

## Flaky Test Policy
- Before trusting a diff-cover gate failure, check whether the test run was flaky.
- Quarantine known-flaky tests with `@pytest.mark.xfail(strict=False, reason='flaky: <describe>')`.
- A gate failure blamed on flakiness must be triaged with an xfail mark, not ignored.
- Remove the xfail once the underlying cause is fixed.

## Mutation Testing
- Line coverage is a quantity signal.
- Once a CPU-heavy module reaches about `85%` branch coverage, evaluate mutation testing with `mutmut` or `cosmic-ray` on that module.
- Do not run mutation testing below `80%` branch coverage; it produces no signal.

## Check Command
```bash
# Current combined coverage:
bash scripts/coverage.sh
uv run --no-sync coverage report | grep TOTAL

# Bump floor after stable climb (edit ci.yml coverage job):
# Change: uv run --no-sync coverage report --fail-under=62
# To:     uv run --no-sync coverage report --fail-under=63
```
