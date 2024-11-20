@echo off

for %%a in (co cb sm cv di wlb) do (
  echo Processing category: %%a
  python 4_summ_eval.py %%a
  taskkill /F /IM python.exe /T
)

echo Done processing all categories.

pause