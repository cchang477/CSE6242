@echo off

for %%a in (cb sm cv di wlb) do (
  echo Processing category: %%a
  python 3_test_eval.py %%a
  taskkill /F /IM python.exe /T
)

echo Done processing all categories.

pause