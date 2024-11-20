@echo off

rem fullset is (co cb sm cv di wlb)

for %%a in (co cb sm cv di wlb) do (
  echo Processing category: %%a
  python 3_finetune.py %%a
  taskkill /F /IM python.exe /T
)

echo Done processing all categories.

pause