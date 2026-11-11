# PROPS project page

This branch holds the built project page and is served by GitHub Pages. Do not edit it by hand. It is built from
the `main` branch with `page/make_page.py`, which reads `leaderboard/board.json` there:

```bash
git worktree add ../PROPS-site gh-pages                  # once, next to a checkout of main
python3 page/make_page.py --out ../PROPS-site             # after every change to the leaderboard
```
