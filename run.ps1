# auto_guitar_tone 실행 런처 — 진행상황(trial별 loss/best)이 이 창에 실시간 표시.
# 사용: 파일 우클릭 → "PowerShell에서 실행", 또는 PowerShell에서  ./run.ps1
# 인자 예: ./run.ps1 -trials 80 -gain 0.3 -stage2trials 0
param(
    [string]$di           = "work/my_di4.wav",   # 4s DI (빠름). 정밀하게는 my_di6.wav
    [string]$target       = "work/target_guitar.wav",
    [int]   $play         = -1,     # -1=자동탐지(Realtek MME 라인아웃). 고정하려면 인덱스 지정
    [double]$gain         = 0.25,
    [int]   $trials       = 100,    # Stage 1 횟수
    [int]   $stage2trials = 50,     # Stage 2 (DELAY/REVERB) 횟수. 0=건너뜀
    [double]$paramdelay   = 0.1,    # Stage B 파라미터 전송 간격 (모델 고정 시 단축)
    [double]$trimdi       = 4.0     # DI 자동 트림 길이(초). 0=전체 사용
)

$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot

# play=-1 means auto-detect: drop the --play-device arg so main/preflight detect it.
$playArgs = @()
$playLabel = "auto-detect"
if ($play -ge 0) { $playArgs = @("--play-device", $play); $playLabel = $play }

Write-Host "=== auto_guitar_tone ===" -ForegroundColor Cyan
Write-Host "DI=$di  target=$target  play=$playLabel  gain=$gain"
Write-Host "trials=$trials  stage2=$stage2trials  param-delay=$paramdelay  trim-di=$trimdi`n"

# preflight
python src/preflight.py --di $di --target $target @playArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nPreflight failed — fix the above and retry." -ForegroundColor Red
    Read-Host "Press Enter to close"; exit 1
}

Write-Host "`n--- optimization start ---`n" -ForegroundColor Green
python src/main.py `
    --di $di `
    --target $target `
    @playArgs `
    --play-gain $gain `
    --trials $trials `
    --stage2-trials $stage2trials `
    --param-delay $paramdelay `
    --trim-di $trimdi

Write-Host "`n=== Done. See work/results/ and work/result.txt ===" -ForegroundColor Cyan
Read-Host "Press Enter to close window"
