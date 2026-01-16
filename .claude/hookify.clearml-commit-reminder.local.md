---
name: clearml-commit-reminder
enabled: true
event: file
pattern: .*clearml.*\.py$|.*pipeline.*controller.*\.py$|.*step_runner.*\.py$
action: warn
---

## Remember to Commit and Push ClearML Changes

When modifying ClearML-related files (pipeline controllers, step runners, etc.), remember:

1. **Test locally first** if possible
2. **Commit your changes**: `git add -A && git commit -m "fix: <description>"`
3. **Push to GitHub**: `git push`
4. **ClearML agents clone from git** - they won't see your changes until pushed!

**Common files that need pushing:**
- `src/dapidl/pipeline/*.py` - Pipeline controllers
- `scripts/clearml_step_runner_*.py` - Step runner scripts
- `src/dapidl/pipeline/steps/*.py` - Individual pipeline steps

**Quick workflow:**
```bash
git add -A && git commit -m "fix: Update ClearML step runners" && git push
```
