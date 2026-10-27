# Releasing

Releases are published by [`.github/workflows/release.yml`](../.github/workflows/release.yml) when a version tag is pushed. The workflow uses PyPI trusted publishing, so no API token is stored anywhere.

## One-time setup

1. **Register the pending publishers.** On [TestPyPI](https://test.pypi.org/manage/account/publishing/) and on [PyPI](https://pypi.org/manage/account/publishing/), add a pending GitHub publisher with these settings:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `reliability-lab` |
   | Owner | `Sathapana` |
   | Repository name | `reliability-lab` |
   | Workflow name | `release.yml` |
   | Environment name | `testpypi` on TestPyPI, `pypi` on PyPI |

2. **Create the two environments.** In the GitHub repository, open Settings → Environments and create `testpypi` and `pypi`. On `pypi`, add yourself as a required reviewer, so each PyPI upload waits for your approval after TestPyPI succeeds.

## Each release

1. Set the version in `src/reliability_lab/_version.py` and move the `Unreleased` entries in `CHANGELOG.md` under that version.
2. Merge to `main` with CI green.
3. Tag the merge commit and push the tag:

   ```bash
   git tag v0.1.0a1
   git push origin v0.1.0a1
   ```

The workflow checks that the tag matches the package version, builds the wheel and sdist, and runs `twine check`. It then publishes to TestPyPI and, after your approval, to PyPI.

Pre-releases such as `0.1.0a1` are installed with `pip install --pre reliability-lab`.
