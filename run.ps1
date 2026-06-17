# auto_guitar_tone 실행 런처 — 진행상황(trial별 loss/best)이 이 창에 실시간 표시.
# 사용: 파일 우클릭 → "PowerShell에서 실행", 또는 PowerShell에서  ./run.ps1
# 인자 바꾸려면:  ./run.ps1 -trials 80 -gain 0.3
param(
    [string]$di      = "work/my_di6.wav",
    [string]$target  = "work/target_guitar.wav",
    [int]   $play    = 7,
    [double]$gain    = 0.25,
    [int]   $trials  = 150
)

$env:PYTHONIOENCODING = "utf-8"           # 한글 진행로그 깨짐 방지
Set-Location $PSScriptRoot                 # 스크립트 위치 = 프로젝트 루트

Write-Host "=== auto_guitar_tone 실행 ===" -ForegroundColor Cyan
Write-Host "DI=$di  target=$target  play-device=$play  gain=$gain  trials=$trials`n"

# 실행 전 사전점검
python src/preflight.py --di $di --target $target --play-device $play
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n사전점검 실패 — 위 항목 해결 후 다시." -ForegroundColor Red
    Read-Host "엔터로 닫기"; exit 1
}

Write-Host "`n--- 최적화 시작 (진행상황 아래 실시간) ---`n" -ForegroundColor Green
python src/main.py --di $di --target $target --play-device $play --play-gain $gain --trials $trials

Write-Host "`n=== 끝. work/result.txt + work/best_reamp.wav 확인 ===" -ForegroundColor Cyan
Read-Host "엔터로 창 닫기"
